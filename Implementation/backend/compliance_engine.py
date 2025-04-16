import json
import os
from datetime import datetime
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

"""Create & Refine Compliance Machine Learning Engine. This engine will find compliance issues in the data.
"""


class ComplianceEngine:

    def __init__(self):

        self.soc2_labels = ['security', 'availability', 'processing_integrity', 'confidentiality', 'privacy',
                            'compliant']

        # File paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.feedback_file = os.path.join(self.base_dir, "models", "feedback", "compliance_feedback.json")
        self.combined_training_file = os.path.join(self.base_dir, "models", "feedback",
                                                   "compliance_trainingdata_with_feedback.txt")
        self.selected_model = "Logistic Regression Model"

    def get_selected_models(self):
        return [self.selected_model]

    def SOC2_classifier(self):
        """Text Classifier for SOC2 compliance."""
        vectorizer_soc2 = CountVectorizer(max_features=1000)
        base_classifiers = [
            LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=0.1,
                penalty='l2',
                solver='saga',
                class_weight='balanced'
            )
        ]
        multi_classifier_soc2 = MultiOutputClassifier(
            estimator=base_classifiers[0],  # This will be replaced for each label
            n_jobs=-1  # Use all available cores
        )
        multi_classifier_soc2.estimators_ = base_classifiers

        return multi_classifier_soc2, vectorizer_soc2,

    def train_SOC2_classifier(self, classifier, vectorizer):
        """Train the chosen classifier on the given data."""
        print(">>> Training classifier...")
        try:
            # Path to the training data file
            file_path = os.path.join(self.base_dir, "SampleData", "compliance_trainingdata.txt")

            print(f">>> Looking for training data at: {file_path}")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Training data file not found at: {file_path}")

            df = pd.read_csv(file_path)

            for label in self.soc2_labels:
                df[label] = pd.to_numeric(df[label])
            x = vectorizer.fit_transform(df['text'])
            y = df[self.soc2_labels].values

            # Train the classifier
            classifier.fit(x, y)
            print(">>> Classifier trained successfully.")

        except Exception as e:
            print(f"An error occurred during training: {e}")
            raise

        print(">>> Saving trained model to file...")
        try:
            self.save_trained_model(classifier, vectorizer, "SOC2_Model")
        except Exception as e:
            print(f"An error occurred while saving the trained model: {e}")

    def save_trained_model(self, classifier, vectorizer, classifier_folder_name):
        """ Save model training to file """

        file_dir = os.path.join(self.base_dir, "models")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        model_dir = os.path.join(file_dir, classifier_folder_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        joblib.dump(vectorizer, f"{model_dir}/vectorizer.joblib")
        joblib.dump(classifier, f"{model_dir}/classifier.joblib")
        print(f">>> Model saved to {model_dir}")

    def load_trained_model(self, classifier_folder_name):
        """ Load model training from file """
        model_dir = os.path.join(self.base_dir, "models", classifier_folder_name)
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
        classifier = joblib.load(os.path.join(model_dir, "classifier.joblib"))
        return classifier, vectorizer

    def save_feedback_for_training(self, violations, feedback):
        """ Save new feedback data to training data """
        print(">>> Saving feedback for training...")
        time_stamp = datetime.now().isoformat()

        # clean up text from escape characters
        cleaned_violations = []
        for prediction in violations:
            cleaned_prediction = prediction.copy()
            if "text_to_review" in cleaned_prediction:
                cleaned_prediction["text_to_review"] = ' '.join(cleaned_prediction["text_to_review"].split())
            cleaned_violations.append(cleaned_prediction)

        cleaned_feedback = {}
        for text, violation_labels in feedback.items():
            cleaned_text = ' '.join(text.split())
            cleaned_feedback[cleaned_text] = violation_labels

        feedback_entry = {
            "time_stamp": time_stamp,
            "model_predictions": cleaned_violations,
            "user_feedback": cleaned_feedback
        }

        try:
            filename = self.feedback_file
            # format file for list of json entries
            with open(filename, 'r') as file:
                # Remove the closing bracket
                content = file.read().rstrip().rstrip(']')

                # If file is empty or doesn't end with [, add opening bracket
                if not content:
                    content = '['

                # Add comma if there's existing content
                if content != '[':
                    content += ','

            # Write back the content plus new entry
            with open(filename, 'w') as file:
                file.write(content)
                file.write('\n')  # Add new line
                json.dump(feedback_entry, file, indent=2)
                file.write('\n]')  # Close the array
                print(f">>> Feedback saved to {filename}")
        except FileNotFoundError:
            print(f">>> Feedback file not found. Creating new file: {filename}")
            file_dir = os.path.join(self.base_dir, "models", "feedback")
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            with open(filename, 'w') as file:
                file.write('[\n')
                json.dump(feedback_entry, file, indent=2)
                file.write('\n]')
        except Exception as e:
            print(f"An error occurred while saving feedback data: {e}")

    def retrain_model_with_new_feedback(self, classifier, vectorizer):
        """ Retrain model with new feedback data """
        print(">>> Retraining model with new feedback...")

        # get existing training data
        existing_training_data_path = os.path.join(self.base_dir, "SampleData", "compliance_trainingdata.txt")
        combined_training_data_path = self.combined_training_file

        try:
            existing_training_data = pd.read_csv(existing_training_data_path)
            filename = self.feedback_file
            with open(filename, 'r') as file:
                feedback_data = json.load(file)
                feedback_texts = []
                feedback_labels = []
                for entry in feedback_data:
                    user_feedback = entry["user_feedback"]
                    for text, violation_labels in user_feedback.items():
                        cleaned_text = ' '.join(text.split())
                        feedback_texts.append(cleaned_text)

                        label_array = [0] * len(self.soc2_labels)
                        for violation in violation_labels:
                            if violation in self.soc2_labels:
                                label_array[self.soc2_labels.index(violation)] = 1

                        feedback_labels.append(label_array)

                # zip feedback and label_array together
                new_data_dict = {
                    'text': feedback_texts
                }
                for i, label in enumerate(self.soc2_labels):
                    new_data_dict[label] = [labels[i] for labels in feedback_labels]

                # ready for combining with training data
                new_data_df = pd.DataFrame(new_data_dict)
                combined_data = pd.concat([existing_training_data, new_data_df], ignore_index=True)

                # save combined training data
                if not os.path.exists(combined_training_data_path):
                    os.makedirs(os.path.dirname(combined_training_data_path), exist_ok=True)

                combined_data.to_csv(combined_training_data_path, index=False)
                print(
                    f">>> Feedback data combined with existing training data and saved to {combined_training_data_path}")

                # create the vectors
                x = vectorizer.fit_transform(combined_data['text'])
                y = combined_data[self.soc2_labels].values

                # train the model
                classifier.fit(x, y)

                # save the model
                self.save_trained_model(classifier, vectorizer, "SOC2_Model")

                print(">>> Retraining model with new feedback complete.")
                return classifier, vectorizer

        except Exception as e:
            print(f"An error occurred while re-training model with feedback data: {e}")
            raise

    def handle_corrections(self, line):

        """Handle user corrections for violations."""

        while True:
            valid_options = self.soc2_labels
            print("\nPlease select from the following: \n" + ", ".join(valid_options))
            user_correction = input("Enter your correction(s): ").lower().strip()

            if user_correction.lower() == 'q':
                print("Exiting review.")
                return None

            # Clean and validate user input
            corrections = [correction.strip() for correction in user_correction.split(",")]

            if all(correction in self.soc2_labels for correction in corrections):
                print("Feedback noted.")
                return corrections
            else:
                print("Invalid correction(s). Please enter a valid option(s).")

    def get_user_feedback(self, line, violations):
        """Get user feedback on the violations found."""

        while True:
            try:
                user_input = input("Enter 'y' to accept finding, 'n' to reject finding, or 'q' to quit: ")

                if user_input.lower() == 'y':
                    print("Feedback noted.")
                    return True, violations
                elif user_input.lower() == 'n':
                    corrections = self.handle_corrections(line)
                    if corrections is None:
                        return False, None
                    return True, corrections
                elif user_input.lower() == 'q':
                    print("Exiting review.")
                    return False, None
                else:
                    print("Invalid input. Please enter 'y', 'n', or 'q'.")
            except Exception as e:
                print(f"An error occurred during feedback: {e}")
            raise

    def predict_violations(self, classifier, vectorizer, line):
        """Predict violations for a given line of text."""
        x = vectorizer.transform([line])

        prediction = classifier.predict(x)
        violations = []

        # Map prediction vector to string labels
        prediction_array = prediction[0]

        for label, violation in enumerate(self.soc2_labels):
            if prediction_array[label] == 1:
                print(f"Violation found: {violation}")
                violations.append(violation)

        # Handle case when no violations are found
        if len(violations) == 0:
            if prediction_array[-1] == 1:  # Check 'compliant' label
                violations.append('compliant')
            else:  # If no violations and not marked compliant, default to compliant
                print("No violations detected, marking as compliant by default")
                violations.append('compliant')

        return violations

    def review_violations(self, classifier, vectorizer, data_to_review):
        """Review violations found by the chosen classifier."""
        print("\nReviewing violations:")
        feedback = {}
        model_predictions = {}

        for line in data_to_review:
            print(f"\nText: {line}")

            # Get predictions
            violations = self.predict_violations(classifier, vectorizer, line)
            model_predictions[line] = violations.copy()

            # Get user feedback
            continue_review, user_feedback = self.get_user_feedback(line, violations)

            if not continue_review:
                break

            if user_feedback is not None:
                feedback[line] = user_feedback

        return model_predictions, feedback

    def ask_user_retrain(self, model_predictions, feedback):
        """Ask user if they want to retrain model with their feedback"""
        while True:
            try:
                save_feedback_question = input("Would you like to retrain the model with your feedback? (y/n): ")
                if save_feedback_question.lower() == 'y':
                    self.save_feedback_for_training(model_predictions, feedback)
                    return True
                elif save_feedback_question.lower() == 'n':
                    print("Feedback not saved.")
                    return False
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

            except Exception as e:
                raise


if __name__ == "__main__":
    print(">>> Running Compliance Engine...")
    compliance_engine = ComplianceEngine()
    # Train model
    soc2_classifier, soc2_vectorizer = compliance_engine.SOC2_classifier()
    compliance_engine.train_SOC2_classifier(soc2_classifier, soc2_vectorizer)

    # Load data to be investigated
    review_data = ["Phone number: 123-456-7890", "Email: billgates@microsoft.com",
                   "He paid with 4532-9256-1499-4387 for the laptop", "My password is 213!Honey",
                   "I love Taylor Swift music!"]
    # Load SOC2 Classifier
    soc2_classifier_loaded, soc2_vectorizer_loaded = compliance_engine.load_trained_model("SOC2_Model")
    # Verify Findings
    model_predictions,feedback = compliance_engine.review_violations(soc2_classifier_loaded, soc2_vectorizer_loaded, review_data)
    # Train model on findings and save model
    answer_retrain = compliance_engine.ask_user_retrain(model_predictions,feedback)
    if answer_retrain:
        compliance_engine.retrain_model_with_new_feedback(soc2_classifier_loaded, soc2_vectorizer_loaded)
    else:
        print("Not retraining model.")
