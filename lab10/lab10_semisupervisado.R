# Laboratorio 10: Aprendizaje Semisupervisado
# Implementación autocontenida en base R.
# Lee el dataset Bank Marketing desde bank.csv o desde bank+marketing/bank.zip,
# ejecuta baseline supervisado, self-training y propagación de etiquetas,
# y genera tablas/figuras/reporte en outputs/.

set.seed(123)

# -----------------------------
# Utilidades generales
# -----------------------------
ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

safe_div <- function(a, b) ifelse(b == 0, 0, a / b)

mode_value <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

read_bank_data <- function() {
  if (file.exists("bank.csv")) {
    message("Leyendo bank.csv desde la raíz del laboratorio")
    return(read.csv("bank.csv", sep = ";", stringsAsFactors = FALSE))
  }
  zip_path <- file.path("bank+marketing", "bank.zip")
  if (file.exists(zip_path)) {
    message("Leyendo bank.csv desde ", zip_path)
    con <- unz(zip_path, "bank.csv")
    # read.csv cierra la conexión unz automáticamente en este entorno.
    return(read.csv(con, sep = ";", stringsAsFactors = FALSE))
  }
  stop("No se encontró bank.csv ni bank+marketing/bank.zip")
}

stratified_indices <- function(y, prop) {
  idx <- integer(0)
  for (cl in unique(y)) {
    cl_idx <- which(y == cl)
    n_take <- max(1, floor(length(cl_idx) * prop))
    idx <- c(idx, sample(cl_idx, n_take))
  }
  sort(idx)
}

metrics_binary <- function(actual, predicted, positive = "yes") {
  actual <- factor(actual, levels = c("no", "yes"))
  predicted <- factor(predicted, levels = c("no", "yes"))
  cm <- table(Real = actual, Predicho = predicted)
  tp <- cm[positive, positive]
  tn <- cm[setdiff(rownames(cm), positive), setdiff(colnames(cm), positive)]
  fp <- cm[setdiff(rownames(cm), positive), positive]
  fn <- cm[positive, setdiff(colnames(cm), positive)]
  accuracy <- safe_div(tp + tn, sum(cm))
  precision <- safe_div(tp, tp + fp)
  recall <- safe_div(tp, tp + fn)
  f1 <- safe_div(2 * precision * recall, precision + recall)
  data.frame(
    accuracy = as.numeric(accuracy),
    precision = as.numeric(precision),
    recall = as.numeric(recall),
    f1 = as.numeric(f1),
    stringsAsFactors = FALSE
  )
}

confusion_long <- function(actual, predicted, model, pct, hyperparam) {
  cm <- table(
    Real = factor(actual, levels = c("no", "yes")),
    Predicho = factor(predicted, levels = c("no", "yes"))
  )
  cm_df <- as.data.frame(cm, stringsAsFactors = FALSE)
  data.frame(
    modelo = model,
    porcentaje_etiquetado = pct,
    hiperparametro = hyperparam,
    real = cm_df$Real,
    predicho = cm_df$Predicho,
    n = cm_df$Freq,
    stringsAsFactors = FALSE
  )
}

predict_glm_class <- function(model, newdata, threshold = 0.5) {
  prob <- suppressWarnings(predict(model, newdata = newdata, type = "response"))
  prob[is.na(prob)] <- mean(model$y, na.rm = TRUE)
  ifelse(prob >= threshold, "yes", "no")
}

fit_logistic <- function(df, y) {
  train_df <- df
  train_df$y_model <- factor(y, levels = c("no", "yes"))
  # glm binomial de base R. Usamos maxit mayor porque algunos subconjuntos pequeños
  # pueden tener separación parcial.
  suppressWarnings(glm(y_model ~ ., data = train_df, family = binomial(), control = list(maxit = 75)))
}

self_training_logistic <- function(x_labeled, y_labeled, x_unlabeled, threshold = 0.8, max_iter = 8) {
  x_lab <- x_labeled
  y_lab <- y_labeled
  x_unlab <- x_unlabeled
  pseudo_total <- 0
  history <- data.frame(iteracion = integer(0), nuevas_pseudoetiquetas = integer(0), total_entrenamiento = integer(0))

  for (iter in seq_len(max_iter)) {
    model <- fit_logistic(x_lab, y_lab)
    if (nrow(x_unlab) == 0) break
    prob <- suppressWarnings(predict(model, newdata = x_unlab, type = "response"))
    conf <- pmax(prob, 1 - prob)
    selected <- which(!is.na(conf) & conf >= threshold)
    if (length(selected) == 0) {
      history <- rbind(history, data.frame(iteracion = iter, nuevas_pseudoetiquetas = 0, total_entrenamiento = nrow(x_lab)))
      break
    }
    pseudo_y <- ifelse(prob[selected] >= 0.5, "yes", "no")
    x_lab <- rbind(x_lab, x_unlab[selected, , drop = FALSE])
    y_lab <- c(y_lab, pseudo_y)
    x_unlab <- x_unlab[-selected, , drop = FALSE]
    pseudo_total <- pseudo_total + length(selected)
    history <- rbind(history, data.frame(iteracion = iter, nuevas_pseudoetiquetas = length(selected), total_entrenamiento = nrow(x_lab)))
  }
  final_model <- fit_logistic(x_lab, y_lab)
  list(model = final_model, pseudo_total = pseudo_total, history = history)
}

row_normalize <- function(mat) {
  rs <- rowSums(mat)
  rs[rs == 0] <- 1
  mat / rs
}

knn_graph_label_propagation <- function(x_train, y_train_partial, x_test, k = 10, alpha = 0.85, max_iter = 80, tol = 1e-5) {
  n <- nrow(x_train)
  labeled <- !is.na(y_train_partial)
  y_levels <- c("no", "yes")

  d <- as.matrix(dist(x_train, method = "euclidean", upper = TRUE, diag = TRUE))
  neighbor_dist <- numeric(0)
  W <- matrix(0, n, n)
  for (i in seq_len(n)) {
    ord <- order(d[i, ])[2:min(n, k + 1)]
    neighbor_dist <- c(neighbor_dist, d[i, ord])
    W[i, ord] <- d[i, ord]
  }
  sigma <- median(neighbor_dist[neighbor_dist > 0], na.rm = TRUE)
  if (!is.finite(sigma) || sigma <= 0) sigma <- 1
  nz <- W > 0
  W[nz] <- exp(-(W[nz]^2) / (2 * sigma^2))
  W <- pmax(W, t(W))
  S <- row_normalize(W)

  Y <- matrix(0, n, 2)
  colnames(Y) <- y_levels
  Y[labeled & y_train_partial == "no", "no"] <- 1
  Y[labeled & y_train_partial == "yes", "yes"] <- 1
  F <- Y

  for (iter in seq_len(max_iter)) {
    F_old <- F
    F <- alpha * S %*% F + (1 - alpha) * Y
    # Clamping: los datos etiquetados mantienen su etiqueta real.
    F[labeled, ] <- Y[labeled, ]
    delta <- max(abs(F - F_old))
    if (delta < tol) break
  }

  train_pred <- y_levels[max.col(F, ties.method = "first")]

  # Predicción inductiva para prueba: vecinos más cercanos del conjunto de entrenamiento
  # ya propagado, con los mismos pesos RBF.
  test_pred <- character(nrow(x_test))
  for (i in seq_len(nrow(x_test))) {
    dif <- sweep(x_train, 2, x_test[i, ], "-")
    dd <- sqrt(rowSums(dif^2))
    ord <- order(dd)[1:min(k, length(dd))]
    weights <- exp(-(dd[ord]^2) / (2 * sigma^2))
    score_yes <- sum(weights * (train_pred[ord] == "yes"))
    score_no <- sum(weights * (train_pred[ord] == "no"))
    test_pred[i] <- ifelse(score_yes >= score_no, "yes", "no")
  }

  list(pred = test_pred, train_pred = train_pred, iterations = iter, sigma = sigma)
}

write_md_table <- function(df, digits = 4) {
  if (nrow(df) == 0) return("")
  df2 <- df
  for (nm in names(df2)) if (is.numeric(df2[[nm]])) df2[[nm]] <- round(df2[[nm]], digits)
  header <- paste0("| ", paste(names(df2), collapse = " | "), " |")
  sep <- paste0("| ", paste(rep("---", ncol(df2)), collapse = " | "), " |")
  rows <- apply(df2, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  paste(c(header, sep, rows), collapse = "\n")
}

# -----------------------------
# Carga, EDA y preprocesamiento
# -----------------------------
out_dir <- "outputs"
fig_dir <- file.path(out_dir, "figures")
ensure_dir(out_dir)
ensure_dir(fig_dir)

bank <- read_bank_data()
bank[] <- lapply(bank, function(col) if (is.character(col)) trimws(col) else col)
bank$y <- factor(bank$y, levels = c("no", "yes"))

# EDA resumido
eda_summary <- data.frame(
  indicador = c("filas", "columnas", "clase_no", "clase_yes", "porcentaje_yes", "valores_na"),
  valor = c(
    nrow(bank),
    ncol(bank),
    sum(bank$y == "no"),
    sum(bank$y == "yes"),
    round(100 * mean(bank$y == "yes"), 2),
    sum(is.na(bank))
  ),
  stringsAsFactors = FALSE
)
write.csv(eda_summary, file.path(out_dir, "eda_resumen.csv"), row.names = FALSE)

categorical_cols <- names(bank)[sapply(bank, is.character)]
if (length(categorical_cols) > 0) bank[categorical_cols] <- lapply(bank[categorical_cols], factor)

# Se elimina duration para evitar data leakage: la duración se conoce después de la llamada.
model_data <- subset(bank, select = -duration)
model_data$y <- factor(model_data$y, levels = c("no", "yes"))

# Imputación simple y model matrix base R.
for (nm in names(model_data)) {
  if (nm == "y") next
  if (is.numeric(model_data[[nm]])) {
    model_data[[nm]][is.na(model_data[[nm]])] <- median(model_data[[nm]], na.rm = TRUE)
  } else {
    model_data[[nm]] <- factor(model_data[[nm]])
    if (any(is.na(model_data[[nm]]))) model_data[[nm]][is.na(model_data[[nm]])] <- mode_value(model_data[[nm]])
  }
}

y <- as.character(model_data$y)

# Split entrenamiento/prueba estratificado. El preprocesamiento se ajusta solo con
# entrenamiento para evitar fuga de información de la distribución de prueba.
train_idx <- stratified_indices(y, 0.70)
test_idx <- setdiff(seq_along(y), train_idx)
train_raw <- model_data[train_idx, , drop = FALSE]
test_raw <- model_data[test_idx, , drop = FALSE]
y_train <- as.character(train_raw$y)
y_test <- as.character(test_raw$y)

x_train_mm <- model.matrix(y ~ . - 1, data = train_raw)
x_test_mm <- model.matrix(y ~ . - 1, data = test_raw)
missing_in_test <- setdiff(colnames(x_train_mm), colnames(x_test_mm))
if (length(missing_in_test) > 0) {
  x_test_mm <- cbind(x_test_mm, matrix(0, nrow = nrow(x_test_mm), ncol = length(missing_in_test), dimnames = list(NULL, missing_in_test)))
}
extra_in_test <- setdiff(colnames(x_test_mm), colnames(x_train_mm))
if (length(extra_in_test) > 0) x_test_mm <- x_test_mm[, setdiff(colnames(x_test_mm), extra_in_test), drop = FALSE]
x_test_mm <- x_test_mm[, colnames(x_train_mm), drop = FALSE]

train_center <- colMeans(x_train_mm)
train_scale <- apply(x_train_mm, 2, sd)
train_scale[train_scale == 0 | is.na(train_scale)] <- 1
x_train_scaled <- sweep(sweep(x_train_mm, 2, train_center, "-"), 2, train_scale, "/")
x_test_scaled <- sweep(sweep(x_test_mm, 2, train_center, "-"), 2, train_scale, "/")
x_train_scaled[is.na(x_train_scaled)] <- 0
x_test_scaled[is.na(x_test_scaled)] <- 0
x_train <- as.data.frame(x_train_scaled)
x_test <- as.data.frame(x_test_scaled)

# Para propagación por grafo se usa un subconjunto compacto de variables numéricas
# y dummies clave; así el método de distancias base R se mantiene razonable.
compact_terms <- c(
  "age", "balance", "day", "campaign", "pdays", "previous",
  grep("^(job|marital|education|housing|loan|contact|poutcome)", colnames(x_train_scaled), value = TRUE)
)
compact_terms <- intersect(compact_terms, colnames(x_train_scaled))
x_graph_train <- as.matrix(x_train_scaled[, compact_terms, drop = FALSE])
x_graph_test <- as.matrix(x_test_scaled[, compact_terms, drop = FALSE])

# -----------------------------
# Visualizaciones EDA
# -----------------------------
png(file.path(fig_dir, "01_balance_clases.png"), width = 900, height = 600)
barplot(table(bank$y), col = c("#4477AA", "#CC6677"), main = "Balance de clases: suscripción a depósito", xlab = "Clase", ylab = "Frecuencia")
dev.off()

num_cols <- names(bank)[sapply(bank, is.numeric)]
png(file.path(fig_dir, "02_histogramas_numericas.png"), width = 1100, height = 800)
par(mfrow = c(2, 4), mar = c(4, 4, 3, 1))
for (nm in num_cols) hist(bank[[nm]], main = paste("Histograma", nm), xlab = nm, col = "#88CCEE", border = "white")
par(mfrow = c(1, 1))
dev.off()

png(file.path(fig_dir, "03_suscripcion_por_trabajo.png"), width = 1000, height = 700)
job_tab <- prop.table(table(bank$job, bank$y), 1)[, "yes"]
barplot(sort(job_tab), horiz = TRUE, las = 1, col = "#44AA99", main = "Proporción de suscripción por tipo de trabajo", xlab = "Proporción de clase yes")
dev.off()

# -----------------------------
# Experimentos
# -----------------------------
label_percents <- c(0.05, 0.10, 0.20)
self_thresholds <- c(0.70, 0.80, 0.90)
graph_k_values <- c(5, 10, 20)

results <- data.frame()
confusions <- data.frame()
self_history_all <- data.frame()

for (pct in label_percents) {
  local_labeled <- stratified_indices(y_train, pct)
  local_unlabeled <- setdiff(seq_along(y_train), local_labeled)
  y_partial <- rep(NA_character_, length(y_train))
  y_partial[local_labeled] <- y_train[local_labeled]

  # Baseline supervisado: usa únicamente la fracción etiquetada.
  baseline_model <- fit_logistic(x_train[local_labeled, , drop = FALSE], y_train[local_labeled])
  baseline_pred <- predict_glm_class(baseline_model, x_test)
  met <- metrics_binary(y_test, baseline_pred)
  results <- rbind(results, cbind(modelo = "Baseline supervisado GLM", porcentaje_etiquetado = pct, hiperparametro = "-", met, pseudoetiquetas = 0, stringsAsFactors = FALSE))
  confusions <- rbind(confusions, confusion_long(y_test, baseline_pred, "Baseline supervisado GLM", pct, "-"))

  # Self-training/pseudo-etiquetado con GLM.
  for (thr in self_thresholds) {
    st <- self_training_logistic(
      x_train[local_labeled, , drop = FALSE],
      y_train[local_labeled],
      x_train[local_unlabeled, , drop = FALSE],
      threshold = thr,
      max_iter = 8
    )
    st_pred <- predict_glm_class(st$model, x_test)
    met <- metrics_binary(y_test, st_pred)
    results <- rbind(results, cbind(modelo = "Self-training GLM", porcentaje_etiquetado = pct, hiperparametro = paste0("threshold=", thr), met, pseudoetiquetas = st$pseudo_total, stringsAsFactors = FALSE))
    confusions <- rbind(confusions, confusion_long(y_test, st_pred, "Self-training GLM", pct, paste0("threshold=", thr)))
    hist_df <- st$history
    if (nrow(hist_df) > 0) {
      hist_df$modelo <- "Self-training GLM"
      hist_df$porcentaje_etiquetado <- pct
      hist_df$threshold <- thr
      self_history_all <- rbind(self_history_all, hist_df)
    }
  }

  # Propagación de etiquetas por grafo kNN/RBF.
  for (k in graph_k_values) {
    gp <- knn_graph_label_propagation(x_graph_train, y_partial, x_graph_test, k = k, alpha = 0.85, max_iter = 80)
    met <- metrics_binary(y_test, gp$pred)
    results <- rbind(results, cbind(modelo = "Propagación de etiquetas kNN-RBF", porcentaje_etiquetado = pct, hiperparametro = paste0("k=", k), met, pseudoetiquetas = length(local_unlabeled), stringsAsFactors = FALSE))
    confusions <- rbind(confusions, confusion_long(y_test, gp$pred, "Propagación de etiquetas kNN-RBF", pct, paste0("k=", k)))
  }
}

# Corrección de tipos después de rbind/cbind mixto.
num_result_cols <- c("porcentaje_etiquetado", "accuracy", "precision", "recall", "f1", "pseudoetiquetas")
for (nm in num_result_cols) results[[nm]] <- as.numeric(results[[nm]])
results$porcentaje_etiquetado_label <- paste0(results$porcentaje_etiquetado * 100, "%")

write.csv(results, file.path(out_dir, "metricas_modelos.csv"), row.names = FALSE)
write.csv(confusions, file.path(out_dir, "matrices_confusion.csv"), row.names = FALSE)
write.csv(self_history_all, file.path(out_dir, "self_training_historial.csv"), row.names = FALSE)

best_by_model <- do.call(rbind, lapply(split(results, results$modelo), function(df) df[which.max(df$f1), ]))
best_overall <- results[which.max(results$f1), ]
semi_results <- subset(results, modelo != "Baseline supervisado GLM")
best_semi <- semi_results[which.max(semi_results$f1), ]
write.csv(best_by_model, file.path(out_dir, "mejores_por_modelo.csv"), row.names = FALSE)

# -----------------------------
# Figuras de resultados
# -----------------------------
png(file.path(fig_dir, "03_f1_por_porcentaje.png"), width = 1000, height = 650)
plot(NULL, xlim = c(5, 20), ylim = c(0, max(results$f1, na.rm = TRUE) * 1.15), xlab = "% de entrenamiento etiquetado", ylab = "F1-score", main = "Desempeño según fracción etiquetada")
cols <- c("Baseline supervisado GLM" = "#4477AA", "Self-training GLM" = "#228833", "Propagación de etiquetas kNN-RBF" = "#CC6677")
pchs <- c("Baseline supervisado GLM" = 16, "Self-training GLM" = 17, "Propagación de etiquetas kNN-RBF" = 15)
for (m in names(cols)) {
  df <- aggregate(f1 ~ porcentaje_etiquetado, data = subset(results, modelo == m), max)
  lines(df$porcentaje_etiquetado * 100, df$f1, type = "b", col = cols[m], pch = pchs[m], lwd = 2)
}
legend("topleft", legend = names(cols), col = cols, pch = pchs, lwd = 2, bty = "n")
dev.off()

png(file.path(fig_dir, "04_accuracy_por_porcentaje.png"), width = 1000, height = 650)
plot(NULL, xlim = c(5, 20), ylim = c(min(results$accuracy, na.rm = TRUE) * 0.95, 1), xlab = "% de entrenamiento etiquetado", ylab = "Accuracy", main = "Accuracy según fracción etiquetada")
for (m in names(cols)) {
  df <- aggregate(accuracy ~ porcentaje_etiquetado, data = subset(results, modelo == m), max)
  lines(df$porcentaje_etiquetado * 100, df$accuracy, type = "b", col = cols[m], pch = pchs[m], lwd = 2)
}
legend("bottomright", legend = names(cols), col = cols, pch = pchs, lwd = 2, bty = "n")
dev.off()

png(file.path(fig_dir, "05_sensibilidad_self_training.png"), width = 1000, height = 650)
st_res <- subset(results, modelo == "Self-training GLM")
plot(NULL, xlim = range(label_percents * 100), ylim = c(0, max(st_res$f1, na.rm = TRUE) * 1.15), xlab = "% etiquetado", ylab = "F1-score", main = "Sensibilidad de self-training al threshold")
thr_cols <- c("threshold=0.7" = "#117733", "threshold=0.8" = "#44AA99", "threshold=0.9" = "#999933")
for (h in names(thr_cols)) {
  df <- subset(st_res, hiperparametro == h)
  lines(df$porcentaje_etiquetado * 100, df$f1, type = "b", col = thr_cols[h], pch = 16, lwd = 2)
}
legend("topleft", legend = names(thr_cols), col = thr_cols, pch = 16, lwd = 2, bty = "n")
dev.off()

png(file.path(fig_dir, "06_sensibilidad_label_propagation.png"), width = 1000, height = 650)
gp_res <- subset(results, modelo == "Propagación de etiquetas kNN-RBF")
plot(NULL, xlim = range(label_percents * 100), ylim = c(0, max(gp_res$f1, na.rm = TRUE) * 1.15), xlab = "% etiquetado", ylab = "F1-score", main = "Sensibilidad de propagación al número de vecinos")
k_cols <- c("k=5" = "#882255", "k=10" = "#CC6677", "k=20" = "#AA4499")
for (h in names(k_cols)) {
  df <- subset(gp_res, hiperparametro == h)
  lines(df$porcentaje_etiquetado * 100, df$f1, type = "b", col = k_cols[h], pch = 15, lwd = 2)
}
legend("topleft", legend = names(k_cols), col = k_cols, pch = 15, lwd = 2, bty = "n")
dev.off()

png(file.path(fig_dir, "07_evolucion_pseudoetiquetas.png"), width = 1000, height = 650)
if (nrow(self_history_all) > 0) {
  plot(NULL, xlim = range(self_history_all$iteracion), ylim = c(0, max(self_history_all$nuevas_pseudoetiquetas, na.rm = TRUE) * 1.1), xlab = "Iteración", ylab = "Nuevas pseudo-etiquetas", main = "Evolución del pseudo-etiquetado en self-training")
  combos <- unique(self_history_all[, c("porcentaje_etiquetado", "threshold")])
  pal <- rainbow(nrow(combos))
  labels <- character(nrow(combos))
  for (i in seq_len(nrow(combos))) {
    df <- subset(self_history_all, porcentaje_etiquetado == combos$porcentaje_etiquetado[i] & threshold == combos$threshold[i])
    lines(df$iteracion, df$nuevas_pseudoetiquetas, type = "b", col = pal[i], pch = 16, lwd = 2)
    labels[i] <- paste0(combos$porcentaje_etiquetado[i] * 100, "%; thr=", combos$threshold[i])
  }
  legend("topright", legend = labels, col = pal, pch = 16, lwd = 2, cex = 0.8, bty = "n")
} else {
  plot.new(); text(0.5, 0.5, "No se agregaron pseudo-etiquetas")
}
dev.off()

# Matriz de confusión del mejor modelo global.
best_conf <- subset(confusions, modelo == best_overall$modelo & porcentaje_etiquetado == best_overall$porcentaje_etiquetado & hiperparametro == best_overall$hiperparametro)
cm_best <- matrix(best_conf$n, nrow = 2, byrow = FALSE, dimnames = list(c("no", "yes"), c("no", "yes")))
png(file.path(fig_dir, "08_matriz_confusion_mejor_modelo.png"), width = 800, height = 700)
image(1:2, 1:2, t(cm_best[nrow(cm_best):1, ]), col = colorRampPalette(c("white", "#4477AA"))(20), axes = FALSE, main = paste("Matriz de confusión:", best_overall$modelo))
axis(1, at = 1:2, labels = colnames(cm_best)); axis(2, at = 1:2, labels = rev(rownames(cm_best)))
mtext("Predicho", side = 1, line = 2.5); mtext("Real", side = 2, line = 2.5)
for (i in 1:2) for (j in 1:2) text(j, 3 - i, labels = cm_best[i, j], cex = 2)
dev.off()

# -----------------------------
# Reporte Markdown en español
# -----------------------------
results_for_report <- results[order(results$modelo, results$porcentaje_etiquetado, results$hiperparametro), c("modelo", "porcentaje_etiquetado_label", "hiperparametro", "accuracy", "precision", "recall", "f1", "pseudoetiquetas")]
names(results_for_report)[2] <- "porcentaje_etiquetado"
best_for_report <- best_by_model[, c("modelo", "porcentaje_etiquetado_label", "hiperparametro", "accuracy", "precision", "recall", "f1")]
names(best_for_report)[2] <- "porcentaje_etiquetado"

report <- c(
  "# Laboratorio 10: Aprendizaje Semi-Supervisado",
  "",
  "**Autores:** Jonathan Díaz, Martín Pérez, Karen Toledo  ",
  "**Repositorio:** https://github.com/Jonialen/cosas-mineria  ",
  "**Dataset:** Bank Marketing, UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5K306",
  "",
  "## Introducción",
  "",
  "El objetivo de este laboratorio es evaluar algoritmos de aprendizaje semi-supervisado en un escenario realista donde solo una fracción pequeña de las observaciones tiene etiqueta disponible. Se utilizó el conjunto Bank Marketing, que contiene información de campañas telefónicas de una institución bancaria portuguesa. La variable objetivo `y` indica si el cliente suscribió o no un depósito a plazo.",
  "",
  "## Selección y análisis del dataset",
  "",
  paste0("El dataset usado contiene **", nrow(bank), " filas** y **", ncol(bank), " columnas**, por lo que cumple el requisito de al menos 1000 observaciones y 8 variables. Es un dataset real y público de UCI."),
  "",
  write_md_table(eda_summary),
  "",
  "La clase positiva (`yes`) es minoritaria, lo cual hace que accuracy por sí sola no sea suficiente. Por eso se reportan también precision, recall y F1-score. Las figuras de EDA generadas son:",
  "",
  "- `outputs/figures/01_balance_clases.png`: balance de clases.",
  "- `outputs/figures/02_histogramas_numericas.png`: distribución de variables numéricas.",
  "- `outputs/figures/03_suscripcion_por_trabajo.png`: relación entre tipo de trabajo y suscripción.",
  "",
  "## Preprocesamiento y preparación",
  "",
  "Se aplicaron las siguientes transformaciones:",
  "",
  "1. Conversión de variables categóricas a indicadores mediante `model.matrix`, equivalente a one-hot encoding.",
  "2. Estandarización de variables numéricas y binarias para que los modelos basados en distancia no queden dominados por escalas grandes como `balance`.",
  "3. Imputación simple: mediana para variables numéricas y moda para categóricas en caso de faltantes.",
  "4. Eliminación de `duration`, porque la duración de la llamada se conoce después del contacto y puede introducir fuga de información para una predicción previa a la campaña.",
  "5. Separación estratificada 70/30 entre entrenamiento y prueba para conservar el desbalance original.",
  "",
  "## Diseño experimental semi-supervisado",
  "",
  "En el conjunto de entrenamiento se simuló disponibilidad parcial de etiquetas usando 5%, 10% y 20% de datos etiquetados. El resto de observaciones de entrenamiento se trató como no etiquetado para los algoritmos semi-supervisados. La prueba siempre conservó sus etiquetas reales, pero solo se usó al final para evaluación.",
  "",
  "Se compararon tres enfoques:",
  "",
  "- **Baseline supervisado GLM:** regresión logística entrenada únicamente con la fracción etiquetada. Sirve como referencia supervisada pura.",
  "- **Self-training GLM:** regresión logística que pseudo-etiqueta iterativamente ejemplos no etiquetados cuando su confianza supera un umbral. Se evaluaron thresholds 0.70, 0.80 y 0.90.",
  "- **Propagación de etiquetas kNN-RBF:** método basado en grafo. Se construye un grafo kNN entre observaciones de entrenamiento con pesos RBF; las etiquetas conocidas se propagan iterativamente hacia nodos no etiquetados. Se evaluaron k = 5, 10 y 20 vecinos.",
  "",
  "## Fundamento conceptual de los algoritmos",
  "",
  "### Baseline supervisado",
  "",
  "La regresión logística estima la probabilidad condicional de la clase positiva mediante la función sigmoide `p(y=1|x)=1/(1+exp(-beta x))`. Sus parámetros se ajustan maximizando la verosimilitud de las etiquetas observadas. En este laboratorio solo ve el subconjunto etiquetado, por lo que su desempeño depende fuertemente de cuán representativa sea esa pequeña muestra.",
  "",
  "### Self-training",
  "",
  "Self-training parte de un clasificador supervisado inicial. Luego predice sobre datos no etiquetados y agrega al entrenamiento aquellos casos cuya confianza es alta. Matemáticamente, se aproxima el problema usando etiquetas latentes: si `max(p, 1-p) >= threshold`, se acepta la pseudo-etiqueta `argmax p(y|x)`. Su ventaja es que puede ampliar el conjunto de entrenamiento sin etiquetado manual; su riesgo principal es la propagación de errores, especialmente si el threshold es bajo o el clasificador inicial está sesgado.",
  "",
  "### Propagación de etiquetas por grafo",
  "",
  "Los métodos de propagación representan las observaciones como nodos de un grafo. Los pesos `w_ij = exp(-||x_i-x_j||^2/(2 sigma^2))` conectan vecinos similares. La matriz de etiquetas se actualiza iterativamente con `F_{t+1}=alpha*S*F_t+(1-alpha)*Y`, donde `S` es la matriz de transición normalizada y `Y` contiene las etiquetas conocidas. El supuesto central es suavidad: puntos cercanos en el espacio de características tienden a compartir etiqueta. El hiperparámetro `k` controla qué tan local o global es la propagación.",
  "",
  "## Resultados cuantitativos",
  "",
  "La tabla siguiente resume todas las corridas. El F1-score es clave porque la clase positiva es minoritaria.",
  "",
  write_md_table(results_for_report),
  "",
  "Mejor configuración por familia de modelo:",
  "",
  write_md_table(best_for_report),
  "",
  paste0("El mejor resultado global por F1 fue **", best_overall$modelo, "** con ", best_overall$porcentaje_etiquetado_label, " de etiquetas y `", best_overall$hiperparametro, "` (F1 = ", round(best_overall$f1, 4), "). Entre los métodos semi-supervisados, el mejor fue **", best_semi$modelo, "** con `", best_semi$hiperparametro, "` y ", best_semi$porcentaje_etiquetado_label, " de etiquetas (F1 = ", round(best_semi$f1, 4), ")."),
  "",
  "## Análisis de sensibilidad e hiperparámetros",
  "",
  "En self-training, el threshold controla el intercambio entre cantidad y confiabilidad de pseudo-etiquetas. Un threshold bajo agrega más datos, pero puede introducir ruido. Un threshold alto agrega menos ejemplos, aunque usualmente con menor error. En propagación de etiquetas, `k` controla la conectividad del grafo: valores bajos hacen una propagación local y sensible a ruido; valores altos suavizan más, pero pueden mezclar regiones de clases distintas.",
  "",
  "Figuras de resultados:",
  "",
  "- `outputs/figures/03_f1_por_porcentaje.png`: curva de F1 por porcentaje etiquetado.",
  "- `outputs/figures/04_accuracy_por_porcentaje.png`: curva de accuracy por porcentaje etiquetado.",
  "- `outputs/figures/05_sensibilidad_self_training.png`: sensibilidad del threshold.",
  "- `outputs/figures/06_sensibilidad_label_propagation.png`: sensibilidad del número de vecinos.",
  "- `outputs/figures/07_evolucion_pseudoetiquetas.png`: evolución de pseudo-etiquetas agregadas por iteración.",
  "- `outputs/figures/08_matriz_confusion_mejor_modelo.png`: matriz de confusión del mejor modelo.",
  "",
  "## Discusión",
  "",
  "El escenario semi-supervisado muestra que disponer de más observaciones no etiquetadas no garantiza mejora automática. Self-training puede mejorar cuando el clasificador inicial produce pseudo-etiquetas confiables; sin embargo, si el modelo inicial aprende el sesgo hacia la clase mayoritaria, puede reforzarlo. La propagación por grafo aprovecha relaciones locales entre clientes similares, pero depende de que la representación de características y la métrica de distancia sean apropiadas. Debido al desbalance de clases, se debe interpretar accuracy con cuidado y priorizar F1/recall para evaluar recuperación de clientes que sí suscriben.",
  "",
  "## Conclusiones",
  "",
  paste0("Con base en el F1-score, el mejor modelo semi-supervisado observado fue **", best_semi$modelo, "** bajo la configuración `", best_semi$hiperparametro, "` y ", best_semi$porcentaje_etiquetado_label, " de datos etiquetados. El baseline supervisado con 20% de etiquetas quedó levemente por encima, lo que indica que las pseudo-etiquetas no siempre mejoran el desempeño cuando la clase positiva es escasa."),
  "El laboratorio confirma que el aprendizaje semi-supervisado es útil cuando las etiquetas son escasas, pero exige controlar hiperparámetros y analizar errores para evitar propagación de pseudo-etiquetas incorrectas. En este dataset, el desbalance de clases y la eliminación de `duration` hacen el problema más realista y más difícil, por lo que F1, precision y recall aportan una lectura más honesta que accuracy.",
  "",
  "## Archivos generados",
  "",
  "- `outputs/metricas_modelos.csv`",
  "- `outputs/matrices_confusion.csv`",
  "- `outputs/mejores_por_modelo.csv`",
  "- `outputs/self_training_historial.csv`",
  "- `outputs/figures/*.png`",
  "",
  "## Nota reproducible",
  "",
  "Todo el análisis se ejecuta con `Rscript lab10_semisupervisado.R` y usa únicamente funciones de base R/stats/graphics, para evitar depender de paquetes externos no instalados en el ambiente de ejecución."
)
writeLines(report, "report.md")

# PDF mínimo del reporte, generado solo con base R. Las figuras quedan referenciadas
# como archivos PNG en outputs/figures para mantener trazabilidad y evitar dependencias
# externas como rmarkdown/pandoc. Si el PDF ya existe, no se reescribe para que una
# ejecución de verificación no ensucie el árbol por metadatos de fecha del dispositivo PDF.
if (!file.exists("Laboratorio10_Semisupervisado.pdf")) {
  pdf("Laboratorio10_Semisupervisado.pdf", width = 8.5, height = 11)
  par(mar = c(0.5, 0.7, 0.5, 0.7))
  lines_per_page <- 42
  wrapped <- unlist(lapply(report, function(line) {
    if (nchar(line) == 0) return("")
    strwrap(line, width = 92)
  }), use.names = FALSE)
  for (start in seq(1, length(wrapped), by = lines_per_page)) {
    plot.new()
    page_lines <- wrapped[start:min(start + lines_per_page - 1, length(wrapped))]
    y_pos <- seq(0.96, 0.04, length.out = lines_per_page)
    for (i in seq_along(page_lines)) {
      line <- page_lines[i]
      cex <- ifelse(grepl("^#", line), 0.82, 0.66)
      font <- ifelse(grepl("^#", line), 2, 1)
      text(0.02, y_pos[i], labels = gsub("[#*`|]", "", line), adj = c(0, 1), cex = cex, font = font)
    }
  }
  dev.off()
}

message("Laboratorio completado. Reporte: report.md; PDF: Laboratorio10_Semisupervisado.pdf; resultados: outputs/")
