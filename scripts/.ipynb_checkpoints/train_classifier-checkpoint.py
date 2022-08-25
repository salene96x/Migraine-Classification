from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import sys
from etl import ETL
import mlflow
#sys.path.insert(0, '/usr/src/')
class Train:
    def __init__(self, config):
        self.config = config
        self.experiment_name = self.config['experiment_name']
    def load_models(self, data: dict):
        models = list()
        for _, model_lib in data.items():
            for k, v in model_lib.items():
                for j in v:
                    models.append(eval(f'{j}()'))
        return models
    def eval_metrics(self, y_true, y_pred, target_names):
        return classification_report(y_true, y_pred, labels=target_names, target_names=target_names, output_dict=True)
    def train(self, models, x_train, x_test, y_train, y_test):
        try :
            current_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            ex_id = current_experiment['experiment_id']
        except:
            ex_id = mlflow.create_experiment()
        labels = self.config['datasets']['labels']
        for j in models:
            mlflow.set_tracking_uri("file:///usr/src/mlruns")
            with mlflow.start_run(experiment_id=ex_id):
                j.fit(x_train, y_train)
                metrics = self.eval_metrics(y_test, j.predict(x_test), labels)
                #metrics = metrics.to_dict()
                for k, v in metrics.items():
                    for x, l in v.items():
                        mlflow.log_metric(f"{k}_{x}", l)
                mlflow.end_run()
    def split_data(self):
        
        etl = ETL()
        df = etl.fit(self.config['datasets']['path'])
        df_x = df[self.datasets['x']]
        df_y = df[self.datasets['y']]
        
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=float(self.config['datasets']['train_test_split']['test_size']), random_state=int(self.config['datasets']['train_test_split']['random_state']))
        
        x_train = scaler_x.fit_tranform(x_train)
        x_test = scaler_x.transform(x_test)
        
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)
        
        return x_train, x_test, y_train, y_test, scaler_x, scaler_y
    def fit(self):
        models = self.load_models(self.config)
        x_train, x_test, y_train, y_test, scaler_x, scaler_y = self.split_data()
        
def read_yaml(FILE_NAME):
    with open(f'/usr/src/configs/{FILE_NAME}', 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return cfg
if __name__ == '__main__':
    try:
        file_name = sys.argv[1]
    except:
        raise ValueError('Missing necessary variable\n-EXPERIMENT_ID')
    
    cfg = read_yaml(file_name)
    trainer = Train(cfg)