

"""
Test different machine learning models for their effectiveness,
accuracy, speed, features and resource requirements.
"""
import os
import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, hamming_loss, make_scorer, confusion_matrix, multilabel_confusion_matrix


class model_evaluation:

    def __init__(self):
        self.soc2_labels = ['security', 'availability', 'processing_integrity', 'confidentiality', 'privacy',
                            'compliant']
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.training_data_path = os.path.join(self.base_dir, "SampleData", "compliance_trainingdata.txt")
        self.cs_results_dir = os.path.join(self.base_dir, "ModelEvaluations")
        if not os.path.exists(self.cs_results_dir):
            os.makedirs(self.cs_results_dir)
    def check_training_data(self):
        # Check distro of training data
        df = pd.read_csv(self.training_data_path)
        print(df.head())

        print("\nLabel distribution:")
        for label in self.soc2_labels:
            df[label] = pd.to_numeric(df[label])
            pos_count = df[label].sum()
            total = len(df)
            print(f"{label}: {pos_count} positive examples ({pos_count / total:.2%})")


    def logistic_regression_model(self):
        """ Evaluate logistic regression model  with training data"""
        # Train model
        df = pd.read_csv(self.training_data_path)
        for label in self.soc2_labels:
            df[label] = pd.to_numeric(df[label])

        vectorizer = CountVectorizer(max_features=1000)
        x = vectorizer.fit_transform(df['text'])
        y = df[self.soc2_labels].values

        # cross validation
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        classifier = MultiOutputClassifier(
            LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear',
        ), n_jobs=-1
        )

        classifier.fit(x_train, y_train)

        # test training
        y_pred = classifier.predict(x_test)
        score = classifier.score(x_test, y_test)
        # Print prediction info for debugging
        # print("\nPrediction information:")
        # print(f"Shape of predictions: {y_pred.shape}")
        # print(f"Unique values in predictions: {np.unique(y_pred)}")
        # print(f"Sum of predictions: {np.sum(y_pred)}")
        # print(f"Model Accuracy Score: {score}")

        accuracy = accuracy_score(y_test, y_pred)

        hamming_loss_score = hamming_loss(y_test, y_pred)

        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f"\n <--- Logistic Regression Model Evaluation --->")
        print(f" Accuracy: {accuracy:.4f}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f" F1: {f1:.4f}")
        print(f" Hamming Loss: {hamming_loss_score:.4f}")

    def random_forest_model(self):
        """ Evaluate random forest model  with training data"""
        df = pd.read_csv(self.training_data_path)
        for label in self.soc2_labels:
            df[label] = pd.to_numeric(df[label])

        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3), #1,3 appears optimal
            min_df=1,
            max_features=1000,
            max_df=0.90, # we want to reduce common words, secrets are going to be unique
        )
        classifier = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                bootstrap=True,
            ),
            n_jobs=-1
        )
        x = vectorizer.fit_transform(df['text'])
        y = df[self.soc2_labels].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        # Print prediction info for debugging
        # print("\nPrediction information for Random Forest debugging:")
        # print(f"Shape of predictions: {y_pred.shape}")
        # print(f"Unique values in predictions: {np.unique(y_pred)}")
        # print(f"Sum of predictions: {np.sum(y_pred)}")

        accuracy = accuracy_score(y_test, y_pred)

        hamming_loss_score = hamming_loss(y_test, y_pred)

        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f"\n <--- Random Forest Model Evaluation --->")
        print(f" Accuracy: {accuracy:.4f}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f" F1: {f1:.4f}")
        print(f" Hamming Loss: {hamming_loss_score:.4f}")

        # Get feature importance
        # feature_names = vectorizer.get_feature_names_out()

    def hyperparameter_tuning(self, x_train, y_train, param_grid, model, name):

        fixed_param_grid = []

        for params_dict in param_grid:
            # Create a new dict for fixed parameters
            fixed_params = {}

            # For each parameter, add the 'estimator__' prefix if it's not already there
            for param, values in params_dict.items():
                if not param.startswith('estimator__'):
                    fixed_params[f'estimator__{param}'] = values
                else:
                    fixed_params[param] = values

            fixed_param_grid.append(fixed_params)

        start = time.time()
        gs_results = GridSearchCV(model, fixed_param_grid, cv=5).fit(x_train, y_train)
        duration = time.time() - start

        if not os.path.exists(self.cs_results_dir):
            os.makedirs(self.cs_results_dir)

        results = pd.DataFrame(gs_results.cv_results_)
        results.loc[:, 'mean_test_score'] *= 100
        results.sort_values(by='rank_test_score', ascending=True, inplace=True)
        try:
            filename = f"cs_results_{name}.csv"
            results.to_csv(os.path.join(self.cs_results_dir, filename))
        except Exception as e:
            print(e)
        return duration, results

    def cross_validate_models(self):
        df = pd.read_csv(self.training_data_path)
        for label in self.soc2_labels:
            df[label] = pd.to_numeric(df[label])

        count_vectorizer = CountVectorizer(max_features=1000)
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # 1,3 appears optimal
            min_df=1,
            max_features=1000,
            max_df=0.90,  # we want to reduce common words, secrets are going to be unique
        )

        x_count = count_vectorizer.fit_transform(df['text'])
        x_tfidf = tfidf_vectorizer.fit_transform(df['text'])
        y = df[self.soc2_labels].values

        models = {
            'LogisticRegression_Count': MultiOutputClassifier(
                LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', solver='liblinear'),
                n_jobs=-1
            ),
            'LogisticRegression_TFIDF': MultiOutputClassifier(
                LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', solver='liblinear'),
                n_jobs=-1
            ),
            'RandomForest_Count': MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10),
                n_jobs=-1
            ),
            'RandomForest_TFIDF': MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10),
                n_jobs=-1
            )
        }

        param_grid_logistic_regression = [{
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 5000]
        }]

        param_grid_random_forest = [{
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }]

        results = {}
        conf_matrix_dict = {}


        highest_accuracy = 0
        most_accurate = ""
        std_accuracy = 0
        lowest_std = 1
        most_consistent = ""
        f1_accuracy = 0
        best_model = None
        for name, model in models.items():
            start_time = time.time()
            x_data = x_count if 'Count' in name else x_tfidf

            f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0) # macro good for infrequent classes
            # Perform 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, x_data, y, cv=kf, scoring=f1_macro_scorer) #geeksforgeeks recommends k=10

            elapsed_time = time.time() - start_time

            # Store and report results

            print(f"\n{name} 5-fold CV Score: F1 Accuracy {scores.mean()*100:.2f}%, Standard Deviation: {scores.std()*100:.2f}% (Time: {elapsed_time:.2f}s)")

            # calculate the highest scoring accuracy model
            if scores.mean() > highest_accuracy:
                highest_accuracy = scores.mean()
                most_accurate = name
                std_accuracy = scores.std()
                best_model = model

            if scores.std() < lowest_std:
                lowest_std = scores.std()
                most_consistent = name
                f1_accuracy = scores.mean()

            # Confusion Matrix
            if 'Count' in name:
                x_confusion = x_count
            else:
                x_confusion = x_tfidf

            X_train, X_test, y_train, y_test = train_test_split(x_confusion, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            confusion_matrix_results = self.confusion_matrix_analysis(y_test, y_pred)

            results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'time': elapsed_time,
                'confusion_matrix': confusion_matrix_results
            }

            if 'LogisticRegression' in name:
                gs_duration, gs_results = self.hyperparameter_tuning(X_train, y_train,param_grid_logistic_regression, model,name)
                print(">>> LR GS Results")
                print(f"Tuning completed in {gs_duration:.2f} seconds")
                results_gs = gs_results.loc[:, ('rank_test_score', 'mean_test_score', 'params')]
                print(results_gs.head(3))
            else:
                gs_duration, gs_results= self.hyperparameter_tuning(X_train, y_train, param_grid_random_forest, model,name)
                print(">>> Random Forest GS Results")
                print(f"Tuning completed in {gs_duration:.2f} seconds")
                results_gs = gs_results.loc[:, ('rank_test_score', 'mean_test_score', 'params')]
                print(results_gs.head(3))

        # print(f"\nThe Scikit model with the highest accuracy is {most_accurate} with a F1 Accuracy score of {highest_accuracy*100:.2f}% with STD of {std_accuracy*100:.2f}%")
        # print(f"\nThe Scikit model with the most consistency is {most_consistent} with a STD score of {lowest_std * 100:.2f}% and a F1 Accuracy of {f1_accuracy*100:.2f}%")

        self.write_report(results)

        return results

    def confusion_matrix_analysis(self, y_true, y_pred):
        """ Show model stats for each SOC2 Label, what is getting false positives and false negatives """

        confusion_mtx = multilabel_confusion_matrix(y_true, y_pred)
        results = {}
        accuracy = None

        for index, label in enumerate(self.soc2_labels):
            tn, fp, fn, tp = confusion_mtx[index].ravel()

            # print(f"\n {label} Results: \n")
            # print(f"True Positives: {tp}")
            # print(f"True Negatives: {tn}")
            # print(f"False Positives: {fp}")
            # print(f"False Negatives: {fn}")

            correct_scores = tp + tn
            incorrect_scores = fp + tn
            total = correct_scores + incorrect_scores

            if total > 0:
                accuracy = correct_scores / total
                #print(f"Accuracy for {label}: {accuracy*100:.2f}%")

            results[label] = {
                "true_positives": str(tp),
                "true_negatives": str(tn),
                "false_positives": str(fp),
                "false_negatives": str(fn),
                "correct_scores": str(correct_scores),
                "incorrect_scores": str(incorrect_scores),
                "accuracy": str(accuracy),
            }

        return results


    def write_report(self, data_strings):
        print(">>>> Writing results to report...")
        try:
            dir_path = os.path.join(self.base_dir,"ModelEvaluations")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            filename = os.path.join(dir_path, "model_report.txt")
            with open(filename, 'w') as file:
                json.dump(data_strings, file, indent=4)
                print(">>> Report written to " + filename)

        except Exception as e:
            print(f" Error writing model evaluation report: {e}")
            raise

if __name__ == "__main__":
    print(">>>> Evaluating......")
    model = model_evaluation()
    model.cross_validate_models()

