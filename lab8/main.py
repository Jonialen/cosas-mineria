import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error
)

warnings.filterwarnings('ignore')
load_dotenv()

SEED = int(os.getenv('RANDOM_SEED', 42))
TRAIN_SIZE = float(os.getenv('TRAIN_SIZE', 0.70))
N_SAMPLES = int(os.getenv('N_SAMPLES', 15000))
PRICE_MIN = int(os.getenv('PRICE_MIN', 0))
PRICE_MAX = int(os.getenv('PRICE_MAX', 1000))

np.random.seed(SEED)

class LabSVM:
    def __init__(self):
        self.data = None
        self.train = None
        self.test = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.train_X_price = None
        self.train_y_price = None
        self.test_X_price = None
        self.test_y_price = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_and_prepare(self):
        """Carga y prepara los datos igual que lab7."""
        print("📥 Cargando datos...")
        pyrrdata = __import__('pyarrow.parquet', fromlist=['parquet'])

        try:
            import rdata
            parsed = rdata.read_rdata('listings.RData')
            self.data = pd.DataFrame(parsed['listings'])
        except:
            print("❌ pyarrow/rdata no disponible. Usando pandas read_parquet...")
            # Alternativa: convertir RData a CSV primero
            print("⚠️  Por favor convertir listings.RData a CSV con:")
            print("   R> load('listings.RData')")
            print("   R> write.csv(listings, 'listings.csv', row.names=FALSE)")
            raise

        print(f"✅ Datos cargados: {self.data.shape}")

    def clean_data(self):
        """Limpia datos igual que lab7."""
        print("\n🧹 Limpiando datos...")

        self.data['price_num'] = pd.to_numeric(
            self.data['price'].str.replace('$', '').str.replace(',', ''),
            errors='coerce'
        )

        self.data['superhost'] = (self.data['host_is_superhost'] == 't').astype(int)
        self.data['instant_book'] = (self.data['instant_bookable'] == 't').astype(int)
        self.data['es_entire_home'] = (self.data['room_type'] == 'Entire home/apt').astype(int)
        self.data['es_private_room'] = (self.data['room_type'] == 'Private room').astype(int)

        features = [
            'accommodates', 'bathrooms', 'bedrooms', 'beds',
            'minimum_nights', 'availability_365', 'number_of_reviews',
            'review_scores_rating', 'review_scores_cleanliness',
            'review_scores_checkin', 'review_scores_communication',
            'review_scores_location', 'review_scores_value',
            'latitude', 'longitude', 'calculated_host_listings_count',
            'reviews_per_month', 'superhost', 'instant_book',
            'es_entire_home', 'es_private_room'
        ]

        datos = self.data[['price_num'] + features].copy()
        datos = datos.dropna(subset=['price_num', 'review_scores_rating',
                                     'bathrooms', 'bedrooms', 'beds', 'reviews_per_month'])
        datos = datos[(datos['price_num'] > PRICE_MIN) & (datos['price_num'] <= PRICE_MAX)]

        print(f"✅ Datos limpios: {datos.shape}")
        self.data = datos
        self.features = features

    def categorize_price(self):
        """Categoriza el precio por cuantiles P33/P66."""
        print("\n📊 Categorizando precios...")
        q33 = self.data['price_num'].quantile(0.33)
        q66 = self.data['price_num'].quantile(0.66)

        print(f"  P33 (barata/media): ${q33:.2f}")
        print(f"  P66 (media/cara):   ${q66:.2f}")

        self.data['precio_cat'] = pd.cut(
            self.data['price_num'],
            bins=[-np.inf, q33, q66, np.inf],
            labels=['barata', 'media', 'cara'],
            right=True
        )

        print(f"\n  Distribución de categorías:")
        for cat in ['barata', 'media', 'cara']:
            count = (self.data['precio_cat'] == cat).sum()
            pct = count / len(self.data) * 100
            print(f"    {cat:10s}: {count:6d} ({pct:5.1f}%)")

    def stratified_split(self):
        """División train/test estratificada igual que lab7."""
        print("\n🔀 División train/test estratificada...")

        np.random.seed(SEED)

        # Muestreo estratificado
        idx_muestra = self.data.groupby('precio_cat', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(N_SAMPLES / 3)), random_state=SEED)
        ).index.tolist()

        datos_sample = self.data.loc[idx_muestra].copy()
        if len(datos_sample) > N_SAMPLES:
            datos_sample = datos_sample.iloc[:N_SAMPLES]

        print(f"  Muestra: {len(datos_sample)} filas")

        # Split 70/30
        np.random.seed(SEED)
        n = len(datos_sample)
        idx_tr = np.random.choice(n, size=int(0.70 * n), replace=False)
        idx_te = np.array([i for i in range(n) if i not in idx_tr])

        self.train = datos_sample.iloc[idx_tr].reset_index(drop=True)
        self.test = datos_sample.iloc[idx_te].reset_index(drop=True)

        print(f"✅ Train: {len(self.train)} | Test: {len(self.test)}")
        print(f"\n  Train - Distribución precio_cat:")
        for cat in ['barata', 'media', 'cara']:
            count = (self.train['precio_cat'] == cat).sum()
            print(f"    {cat}: {count}")

    def prepare_classification(self):
        """Prepara datos para clasificación."""
        print("\n📦 Preparando datos para clasificación...")

        X_train = self.train[self.features].copy()
        X_test = self.test[self.features].copy()
        y_train = self.train['precio_cat'].copy()
        y_test = self.test['precio_cat'].copy()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.train_X = pd.DataFrame(X_train_scaled, columns=self.features)
        self.test_X = pd.DataFrame(X_test_scaled, columns=self.features)
        self.train_y = y_train
        self.test_y = y_test

        print(f"✅ Train X: {self.train_X.shape}, Train y: {self.train_y.shape}")

    def prepare_regression(self):
        """Prepara datos para regresión."""
        print("\n📦 Preparando datos para regresión...")

        X_train = self.train[self.features].copy()
        X_test = self.test[self.features].copy()
        y_train = self.train['price_num'].copy()
        y_test = self.test['price_num'].copy()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.train_X_price = pd.DataFrame(X_train_scaled, columns=self.features)
        self.test_X_price = pd.DataFrame(X_test_scaled, columns=self.features)
        self.train_y_price = y_train
        self.test_y_price = y_test

        print(f"✅ Train X: {self.train_X_price.shape}, Train y: {self.train_y_price.shape}")

    def train_svm_classifiers(self):
        """Entrena múltiples modelos SVM con diferentes kernels."""
        print("\n🤖 Entrenando modelos SVM para clasificación...")

        kernels = ['linear', 'rbf', 'poly']
        svm_models = {}

        for kernel in kernels:
            print(f"\n  Kernel: {kernel}")
            for c_val in [0.1, 1, 10]:
                for gamma_val in ['scale', 0.001, 0.01]:
                    model_name = f'SVM_{kernel}_C{c_val}_G{gamma_val}'

                    svm = SVC(kernel=kernel, C=c_val, gamma=gamma_val,
                              random_state=SEED, probability=True)
                    svm.fit(self.train_X, self.train_y)

                    train_acc = svm.score(self.train_X, self.train_y)
                    test_acc = svm.score(self.test_X, self.test_y)

                    svm_models[model_name] = {
                        'model': svm,
                        'kernel': kernel,
                        'C': c_val,
                        'gamma': gamma_val,
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'overfit': train_acc - test_acc
                    }

                    print(f"    {model_name}: Train={train_acc:.4f}, Test={test_acc:.4f}, Overfit={train_acc-test_acc:.4f}")

        self.results['svm_classifiers'] = svm_models
        return svm_models

    def train_baseline_classifiers(self):
        """Entrena modelos baseline para comparación."""
        print("\n🔬 Entrenando modelos baseline para clasificación...")

        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=SEED),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED),
            'GaussianNB': GaussianNB(),
            'KNN_5': KNeighborsClassifier(n_neighbors=5),
            'KNN_10': KNeighborsClassifier(n_neighbors=10),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=SEED)
        }

        baseline_results = {}
        for name, model in models.items():
            print(f"  {name}...", end=' ')
            model.fit(self.train_X, self.train_y)
            train_acc = model.score(self.train_X, self.train_y)
            test_acc = model.score(self.test_X, self.test_y)
            baseline_results[name] = {
                'model': model,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfit': train_acc - test_acc
            }
            print(f"Train={train_acc:.4f}, Test={test_acc:.4f}")

        self.results['baseline_classifiers'] = baseline_results
        return baseline_results

    def train_svm_regressors(self):
        """Entrena modelos SVM para regresión."""
        print("\n📈 Entrenando modelos SVM para regresión...")

        svm_regressors = {}

        for kernel in ['linear', 'rbf', 'poly']:
            print(f"\n  Kernel: {kernel}")
            for c_val in [0.1, 1, 10]:
                model_name = f'SVR_{kernel}_C{c_val}'

                svr = SVR(kernel=kernel, C=c_val, gamma='scale')
                svr.fit(self.train_X_price, self.train_y_price)

                train_r2 = svr.score(self.train_X_price, self.train_y_price)
                test_r2 = svr.score(self.test_X_price, self.test_y_price)

                train_pred = svr.predict(self.train_X_price)
                test_pred = svr.predict(self.test_X_price)

                train_mse = mean_squared_error(self.train_y_price, train_pred)
                test_mse = mean_squared_error(self.test_y_price, test_pred)

                svm_regressors[model_name] = {
                    'model': svr,
                    'kernel': kernel,
                    'C': c_val,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'overfit': train_r2 - test_r2
                }

                print(f"    {model_name}: Train_R2={train_r2:.4f}, Test_R2={test_r2:.4f}")

        self.results['svm_regressors'] = svm_regressors
        return svm_regressors

    def train_baseline_regressors(self):
        """Entrena modelos baseline para regresión."""
        print("\n📊 Entrenando modelos baseline para regresión...")

        models = {
            'DecisionTree': DecisionTreeRegressor(random_state=SEED),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=SEED),
            'KNN_5': KNeighborsRegressor(n_neighbors=5),
            'LinearRegression': LinearRegression()
        }

        baseline_results = {}
        for name, model in models.items():
            print(f"  {name}...", end=' ')
            model.fit(self.train_X_price, self.train_y_price)
            train_r2 = model.score(self.train_X_price, self.train_y_price)
            test_r2 = model.score(self.test_X_price, self.test_y_price)

            train_pred = model.predict(self.train_X_price)
            test_pred = model.predict(self.test_X_price)
            train_mse = mean_squared_error(self.train_y_price, train_pred)
            test_mse = mean_squared_error(self.test_y_price, test_pred)

            baseline_results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'overfit': train_r2 - test_r2
            }
            print(f"Train_R2={train_r2:.4f}, Test_R2={test_r2:.4f}")

        self.results['baseline_regressors'] = baseline_results
        return baseline_results

    def generate_confusion_matrices(self):
        """Genera matrices de confusión para los mejores modelos."""
        print("\n📋 Generando matrices de confusión...")

        # Encontrar mejor SVM clasificador
        svm_classifiers = self.results['svm_classifiers']
        best_svm = max(svm_classifiers.items(), key=lambda x: x[1]['test_acc'])
        best_svm_name, best_svm_data = best_svm

        # Encontrar mejor baseline
        baseline = self.results['baseline_classifiers']
        best_baseline = max(baseline.items(), key=lambda x: x[1]['test_acc'])
        best_baseline_name, best_baseline_data = best_baseline

        print(f"\n  Mejor SVM: {best_svm_name} (Test Acc: {best_svm_data['test_acc']:.4f})")
        print(f"  Mejor Baseline: {best_baseline_name} (Test Acc: {best_baseline_data['test_acc']:.4f})")

        # Generar matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # SVM
        y_pred_svm = best_svm_data['model'].predict(self.test_X)
        cm_svm = confusion_matrix(self.test_y, y_pred_svm, labels=['barata', 'media', 'cara'])
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['barata', 'media', 'cara'],
                    yticklabels=['barata', 'media', 'cara'])
        axes[0].set_title(f'Matriz de Confusión - {best_svm_name}')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicción')

        # Baseline
        y_pred_baseline = best_baseline_data['model'].predict(self.test_X)
        cm_baseline = confusion_matrix(self.test_y, y_pred_baseline, labels=['barata', 'media', 'cara'])
        sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=['barata', 'media', 'cara'],
                    yticklabels=['barata', 'media', 'cara'])
        axes[1].set_title(f'Matriz de Confusión - {best_baseline_name}')
        axes[1].set_ylabel('Actual')
        axes[1].set_xlabel('Predicción')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
        print("  ✅ Guardado: confusion_matrices.png")

        self.results['best_svm_classifier'] = (best_svm_name, best_svm_data)
        self.results['best_baseline_classifier'] = (best_baseline_name, best_baseline_data)

    def generate_report(self):
        """Genera un informe resumido."""
        print("\n📄 Generando informe...")

        report = []
        report.append("=" * 80)
        report.append("LABORATORIO 8: MÁQUINAS VECTORIALES DE SOPORTE (SVM)")
        report.append("SmartStay Advisors - Predicción de Precios en Airbnb")
        report.append("=" * 80)
        report.append("")

        report.append("📊 DATOS")
        report.append(f"  Muestras totales: {N_SAMPLES}")
        report.append(f"  Train: {len(self.train)} | Test: {len(self.test)}")
        report.append(f"  Features: {len(self.features)}")
        report.append("")

        report.append("🎯 CLASIFICACIÓN - Mejor SVM vs Baseline")
        best_svm_name, best_svm_data = self.results['best_svm_classifier']
        best_baseline_name, best_baseline_data = self.results['best_baseline_classifier']

        report.append(f"\n  SVM: {best_svm_name}")
        report.append(f"    Train Accuracy: {best_svm_data['train_acc']:.4f}")
        report.append(f"    Test Accuracy:  {best_svm_data['test_acc']:.4f}")
        report.append(f"    Overfitting:    {best_svm_data['overfit']:.4f}")

        report.append(f"\n  Baseline: {best_baseline_name}")
        report.append(f"    Train Accuracy: {best_baseline_data['train_acc']:.4f}")
        report.append(f"    Test Accuracy:  {best_baseline_data['test_acc']:.4f}")
        report.append(f"    Overfitting:    {best_baseline_data['overfit']:.4f}")

        report.append("")
        report.append("📈 REGRESIÓN - Mejor SVM vs Baseline")
        svm_regs = self.results['svm_regressors']
        best_svm_reg = max(svm_regs.items(), key=lambda x: x[1]['test_r2'])
        baseline_regs = self.results['baseline_regressors']
        best_baseline_reg = max(baseline_regs.items(), key=lambda x: x[1]['test_r2'])

        report.append(f"\n  SVM: {best_svm_reg[0]}")
        report.append(f"    Train R²:  {best_svm_reg[1]['train_r2']:.4f}")
        report.append(f"    Test R²:   {best_svm_reg[1]['test_r2']:.4f}")
        report.append(f"    Test MSE:  {best_svm_reg[1]['test_mse']:.4f}")

        report.append(f"\n  Baseline: {best_baseline_reg[0]}")
        report.append(f"    Train R²:  {best_baseline_reg[1]['train_r2']:.4f}")
        report.append(f"    Test R²:   {best_baseline_reg[1]['test_r2']:.4f}")
        report.append(f"    Test MSE:  {best_baseline_reg[1]['test_mse']:.4f}")

        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)
        print(report_text)

        with open('informe_lab8.txt', 'w') as f:
            f.write(report_text)

        print("\n✅ Guardado: informe_lab8.txt")

    def run(self):
        """Ejecuta el pipeline completo."""
        print("\n" + "="*80)
        print("LABORATORIO 8 - SVM: CLASIFICACIÓN Y REGRESIÓN")
        print("="*80)

        try:
            self.load_and_prepare()
            self.clean_data()
            self.categorize_price()
            self.stratified_split()
            self.prepare_classification()
            self.prepare_regression()
            self.train_svm_classifiers()
            self.train_baseline_classifiers()
            self.train_svm_regressors()
            self.train_baseline_regressors()
            self.generate_confusion_matrices()
            self.generate_report()

            print("\n✅ ¡Laboratorio completado!")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    lab = LabSVM()
    lab.run()

if __name__ == "__main__":
    main()
