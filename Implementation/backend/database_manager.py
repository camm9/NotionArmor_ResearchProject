import os
import sqlite3
from datetime import datetime


class DatabaseManager:
    """
    Manages database operations.
    - Storing of violation data
    - Storing of model performance data
    - Recording violation review data
    - Retrieval of dashboard stats
    """

    def __init__(self):
        # set up database path
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_dir,"databases","performance.db")
        self.db_dir_path = os.path.join(self.base_dir, "databases")

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.initialize_performance_db()

    def initialize_performance_db(self):
        """Create Database and Tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create review performance data table
        cursor.execute(
            ''' 
            CREATE TABLE IF NOT EXISTS review_data (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            pages_scanned INTEGER,
            total_violations INTEGER,
            status_code TEXT,
            model TEXT DEFAULT "N/A"
            )
            '''
        )

        # Create violation stats table, user_corrected default is False
        cursor.execute(
            ''' 
            CREATE TABLE IF NOT EXISTS violation_data (
                violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                page_id INTEGER,
                page_title TEXT,
                page_url TEXT,
                violation_type TEXT,
                text TEXT,
                user_corrected BOOLEAN DEFAULT 0,
                FOREIGN KEY (review_id) REFERENCES review_data(review_id) 
            )
            '''
        )

        # Create model performance stats table, retraining_performance is FALSE by default
        cursor.execute(
            ''' 
            CREATE TABLE IF NOT EXISTS model_performance (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                timestamp TEXT NOT NULL,
                total_predictions INTEGER,
                corrected_predictions INTEGER,
                model_version TEXT,
                retraining_performance BOOLEAN DEFAULT 0,
                FOREIGN KEY (review_id) REFERENCES review_data(review_id) 
            )
            '''
        )

        # create table to track redactions from review_violations.py
        cursor.execute(
            ''' 
            CREATE TABLE IF NOT EXISTS redaction_data (
            redaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER,
            page_id TEXT,
            page_title TEXT,
            original_text TEXT,
            masked_text TEXT,
            block_ids TEXT,
            applied_to_notion_success BOOLEAN DEFAULT 0,
            timestamp TEXT,
            FOREIGN KEY (review_id) REFERENCES review_data(review_id) 
            )
            '''
        )

        # table to track redactions from ai remediation suggestions
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS ai_remediation_data (
                remediation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                page_id TEXT,
                page_url TEXT,
                original_text TEXT,
                masked_text TEXT,
                block_ids TEXT,
                applied_to_notion_success BOOLEAN DEFAULT 0,
                timestamp TEXT,
                FOREIGN KEY (review_id) REFERENCES review_data(review_id)
            )
            '''
        )

        # Create a table for AI analysis stats
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS ai_analysis_data (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                timestamp TEXT NOT NULL,
                violation_text TEXT,
                categories TEXT,
                severity TEXT,
                page_id TEXT,
                explanation TEXT,
                FOREIGN KEY (review_id) REFERENCES review_data(review_id) 
            )
            '''
        )

        # create table for tracking verification stats of AI performance
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS ai_verification_stats (
                verification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                timestamp TEXT NOT NULL,
                original_categories TEXT,
                modified_categories TEXT,
                original_severity TEXT,
                modified_severity TEXT,
                marked_compliant BOOLEAN DEFAULT 0,
                unchanged BOOLEAN DEFAULT 0,
                category_modified BOOLEAN DEFAULT 0,
                severity_modified BOOLEAN DEFAULT 0,
                FOREIGN KEY (review_id) REFERENCES review_data(review_id)
            )
            '''
        )

        # create table for tracking verification stats of scikit learn models
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS scikit_verification_stats (
                verification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                timestamp TEXT NOT NULL,
                original_categories TEXT,
                modified_categories TEXT,
                marked_compliant BOOLEAN DEFAULT 0,
                unchanged BOOLEAN DEFAULT 0,
                category_modified BOOLEAN DEFAULT 0,
                FOREIGN KEY (review_id) REFERENCES review_data(review_id)
            )
            '''
        )

        conn.commit()
        conn.close()

    def record_violation_review(self, pages_scanned, total_violations, status_code="completed", model="N/A"):
        """ Record a violation review """
        timestamp = datetime.utcnow().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO review_data (timestamp, pages_scanned, total_violations, status_code, model)
        VALUES (?, ?, ?, ?,?)
        ''',
                       (timestamp, pages_scanned, total_violations, status_code, model)
        )

        review_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return review_id

    def record_user_redactions(self, review_id, page_id, page_title, original_text, masked_text, block_ids, applied_to_notion_success):
        """ Record a user redaction from review_violations.py """
        timestamp = datetime.utcnow().isoformat()
        block_ids_comma_string = ",".join(block_ids) if block_ids else ""

        if applied_to_notion_success:
            applied_to_notion_success = 1
        else:
            applied_to_notion_success = 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO redaction_data (review_id, page_id, page_title, original_text, masked_text, block_ids, applied_to_notion_success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (review_id, page_id, page_title, original_text, masked_text, block_ids_comma_string, applied_to_notion_success, timestamp)
        )

        conn.commit()
        conn.close()

    def record_ai_remediation(self, review_id, page_id, page_url, original_text, masked_text, block_ids, applied_to_notion_success):
        """" Record ai remediation from ai_review_violations.py """
        timestamp = datetime.utcnow().isoformat()
        block_ids_comma_string = ",".join(block_ids) if block_ids else ""

        if applied_to_notion_success:
            applied_to_notion_success = 1
        else:
            applied_to_notion_success = 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO ai_remediation_data (review_id, page_id, page_url, original_text, masked_text, block_ids, applied_to_notion_success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (review_id, page_id, page_url, original_text, masked_text, block_ids_comma_string,
                  applied_to_notion_success, timestamp)
        )

        conn.commit()
        conn.close()

    def get_review_summary(self):
        """ Get summary statistics of all reviews"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total Number of Scans Recorded
        cursor.execute(
            'SELECT COUNT(*) as review_count FROM review_data'
        )
        result = cursor.fetchone()
        review_count = result['review_count'] if result else 0

        # Total Number of Violations Found
        cursor.execute(
            'SELECT SUM(total_violations) as violation_count FROM review_data'
        )
        result = cursor.fetchone()
        violation_count = result['violation_count'] if result else 0

        # Number of User Redactions Made to Notion, only count if successful
        cursor.execute(
            'SELECT COUNT(*) as successful_user_redactions FROM redaction_data WHERE applied_to_notion_success = 1'
        )
        result = cursor.fetchone()
        successful_user_redactions = result['successful_user_redactions'] if result else 0

        cursor.execute(
            'SELECT COUNT(*) as unsuccessful_user_redactions FROM redaction_data WHERE applied_to_notion_success = 0'
        )
        result = cursor.fetchone()
        unsuccessful_user_redactions = result['unsuccessful_user_redactions'] if result else 0

        # get number of AI remediations made to Notion
        cursor.execute(
            'SELECT COUNT(*) as successful_ai_remediations FROM ai_remediation_data WHERE applied_to_notion_success = 1'
        )
        result = cursor.fetchone()
        successful_ai_remediations = result['successful_ai_remediations'] if result else 0

        cursor.execute(
            'SELECT COUNT(*) as unsuccessful_ai_remediations FROM ai_remediation_data WHERE applied_to_notion_success = 0'
        )
        result = cursor.fetchone()
        unsuccessful_ai_remediations = result['unsuccessful_ai_remediations'] if result else 0

        conn.close()

        return {"review_count": review_count,
                "violation_count": violation_count,
                "successful_user_redactions": successful_user_redactions,
                "unsuccessful_user_redactions": unsuccessful_user_redactions,
                "successful_ai_remediations": successful_ai_remediations,
                "unsuccessful_ai_remediations": unsuccessful_ai_remediations
                }

    def record_ai_analysis(self, review_id, ai_analysis_results):
        """ Record an ai analysis from ai_violations_review.py """
        timestamp = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for result in ai_analysis_results:
            categories_list = result.get('categories', [])
            categories_str = ','.join(categories_list) if categories_list else ''
            cursor.execute(
                '''
                INSERT INTO ai_analysis_data (review_id, timestamp, violation_text, categories, severity, page_id, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (review_id, timestamp, result.get("text", ""), categories_str,
                 result.get("severity", ""), result.get("location", ""),
                 result.get("explanation", ""))
            )

        conn.commit()
        conn.close()
        print(f">>> Successfully saved {len(ai_analysis_results)} AI analysis results")
        return True

    def get_ai_analysis_stats(self, notion_manager):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total Violations Recorded by AI, each entry is a different violation thus we need to count db entries as violations
        cursor.execute(
            'SELECT COUNT(*) as violation_count FROM ai_analysis_data '
        )
        result = cursor.fetchone()
        violation_count = result['violation_count'] if result else 0

        # Number of Violations Found by Severity
        cursor.execute(
            'SELECT severity, COUNT(*) as count FROM ai_analysis_data GROUP BY severity'
        )
        severity_counts = {}
        for row in cursor.fetchall():
            severity_counts[row['severity']] = row['count']

        # Number of Violations Found by Category
        category_counts = {}
        cursor.execute(
            'SELECT categories FROM ai_analysis_data'
        )
        for row in cursor.fetchall():
            for category in row['categories'].split(','):
                category_counts[category] = category_counts.get(category, 0) + 1


        # Pages with the most violations
        cursor.execute(
            'SELECT page_id, COUNT(*) as count FROM ai_analysis_data GROUP BY page_id ORDER BY count DESC'
        )
        page_counts = {}
        page_ids = []
        for row in cursor.fetchall():
            page_id = row['page_id']
            count = row['count']
            page_counts[page_id] = count
            page_ids.append(page_id)

        conn.close()

        # for pages with most violations, get url
        try:
            page_urls = {}

            for page_id in page_ids:
                try:
                    if page_id and page_id.strip():
                        url = notion_manager.get_page_url(page_id)
                        page_title = notion_manager.get_page_title(page_id)
                        display_value = f"{page_title} ({url})"
                        # user-friendly display with URL and count
                        page_urls[display_value] = page_counts[page_id]
                    else:
                        page_urls[f"Unknown Page"] = page_counts[page_id]
                except Exception as e:
                    print(f">>> Error fetching URL for page {page_id}: {e}")
                    # just the ID if URL fetch fails
                    page_urls[f"Page {page_id}"] = page_counts[page_id]

        except Exception as e:
            print(f">>> Error using NotionManger in database_manager.py: {e}")
            page_urls = page_counts
        return {"violation_count": violation_count,
                "category_counts": category_counts,
                "severity_counts": severity_counts,
                "page_counts": page_urls}

    def record_user_ai_verification_stats(self, review_id, original_categories,modified_categories,original_severity, modified_severity):
        """ Get stats regarding user's verification of AI review findings"""
        timestamp = datetime.utcnow().isoformat()

        # format lists to strings
        original_categories_str = ','.join(original_categories) if isinstance(original_categories,list) else original_categories
        modified_categories_str = ','.join(modified_categories) if isinstance(modified_categories,list) else modified_categories

        # false positive
        marked_compliant = 'compliant' in modified_categories and 'compliant' not in original_categories

        # true positive/negatives
        unchanged = (original_categories_str == modified_categories_str and original_severity == modified_severity)

        # partial true positive/negatives, categories & severities were wrongly assigned
        category_modified = original_categories_str != modified_categories_str and not marked_compliant
        severity_modified = original_severity != modified_severity

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO ai_verification_stats (
                review_id, timestamp, original_categories, modified_categories,
                original_severity, modified_severity, marked_compliant, unchanged,
                category_modified, severity_modified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (review_id, timestamp, original_categories_str, modified_categories_str,
             original_severity, modified_severity, marked_compliant, unchanged,
             category_modified, severity_modified)
        )

        conn.commit()
        conn.close()

        print(">>> Verification stats recorded of AI review findings")
        return True

    def get_ai_verification_stats(self):
        """Get the stats of the AI user verifications"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total user verifications saved
        cursor.execute("SELECT COUNT(*) as total_verfications FROM ai_verification_stats")
        result = cursor.fetchone()
        total_verfications = result['total_verfications'] if result else 0

        # count false positives
        cursor.execute("SELECT COUNT(*) as marked_compliant FROM ai_verification_stats WHERE marked_compliant=1")
        false_positive = cursor.fetchone()['marked_compliant']

        # true positives/negatives, correct prediction
        cursor.execute("SELECT COUNT(*) as unchanged FROM ai_verification_stats WHERE unchanged=1")
        correct_prediction = cursor.fetchone()['unchanged']

        #partial correct, category or severity incorrect
        cursor.execute("SELECT COUNT(*) as category_modified FROM ai_verification_stats WHERE category_modified=1")
        partial_correct_prediction = cursor.fetchone()['category_modified']

        cursor.execute("SELECT COUNT(*) as severity_modified FROM ai_verification_stats WHERE severity_modified=1")
        incorrect_severity = cursor.fetchone()['severity_modified']

        # get percentages
        if total_verfications > 0:
            false_positive_percentage = round((false_positive/total_verfications)*100, 2)
            correct_prediction_percentage = round((correct_prediction/total_verfications)*100, 2)
            partial_correct_prediction_percentage = round((partial_correct_prediction/total_verfications)*100, 2)
            incorrect_severity_percentage = round((incorrect_severity/total_verfications)*100, 2)
        else:
            false_positive_percentage = correct_prediction_percentage = partial_correct_prediction_percentage = incorrect_severity_percentage = 0

        conn.close()

        ai_verification_stats = {
            "total_verifications": total_verfications,
            "false_positives": {
                "count": false_positive,
                "percentage": false_positive_percentage,
            },
            "correct_predictions": {
                "count": correct_prediction,
                "percentage": correct_prediction_percentage,
            },
            "partially_correct_predictions": {
                "count": partial_correct_prediction,
                "percentage": partial_correct_prediction_percentage,
            },
            "severity_modified": {
                "count": incorrect_severity,
                "percentage": incorrect_severity_percentage,
            }
        }

        return ai_verification_stats

    def record_user_model_verification_stats(self, review_id, original_categories,modified_categories):
        """ Get stats regarding user's verification of Scikit Learn Models review findings"""
        timestamp = datetime.utcnow().isoformat()

        # format lists to strings
        original_categories_str = ','.join(original_categories) if isinstance(original_categories,list) else original_categories
        modified_categories_str = ','.join(modified_categories) if isinstance(modified_categories,list) else modified_categories

        # false positive
        marked_compliant = 'compliant' in modified_categories and 'compliant' not in original_categories

        # true positive/negatives
        unchanged = (original_categories_str == modified_categories_str)

        # partial true positive/negatives, categories & severities were wrongly assigned
        category_modified = original_categories_str != modified_categories_str and not marked_compliant

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO scikit_verification_stats (
                review_id, timestamp, original_categories, modified_categories,marked_compliant, unchanged,
                category_modified
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (review_id, timestamp, original_categories_str, modified_categories_str, marked_compliant, unchanged,
             category_modified)
        )

        conn.commit()
        conn.close()

        print(">>> Verification stats recorded of user Scikit Learn review findings")
        return True

    def get_scikit_verification_stats(self):
        """Get the stats of the scikit-learn user verifications"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total user verifications saved
        cursor.execute("SELECT COUNT(*) as total_verifications FROM scikit_verification_stats")
        result = cursor.fetchone()
        total_verifications = result['total_verifications'] if result else 0

        # count false positives
        cursor.execute("SELECT COUNT(*) as marked_compliant FROM scikit_verification_stats WHERE marked_compliant=1")
        false_positive = cursor.fetchone()['marked_compliant']

        # true positives/negatives, correct prediction
        cursor.execute("SELECT COUNT(*) as unchanged FROM scikit_verification_stats WHERE unchanged=1")
        correct_prediction = cursor.fetchone()['unchanged']

        # partial correct, category incorrect
        cursor.execute("SELECT COUNT(*) as category_modified FROM scikit_verification_stats WHERE category_modified=1")
        partial_correct_prediction = cursor.fetchone()['category_modified']

        # get percentages
        if total_verifications > 0:
            false_positive_percentage = round((false_positive / total_verifications) * 100, 2)
            correct_prediction_percentage = round((correct_prediction / total_verifications) * 100, 2)
            partial_correct_prediction_percentage = round((partial_correct_prediction / total_verifications) * 100, 2)
        else:
            false_positive_percentage = correct_prediction_percentage = partial_correct_prediction_percentage = 0

        conn.close()

        scikit_verification_stats = {
            "total_verifications": total_verifications,
            "false_positives": {
                "count": false_positive,
                "percentage": false_positive_percentage,
            },
            "correct_predictions": {
                "count": correct_prediction,
                "percentage": correct_prediction_percentage,
            },
            "partially_correct_predictions": {
                "count": partial_correct_prediction,
                "percentage": partial_correct_prediction_percentage,
            }
        }

        return scikit_verification_stats

    def get_logistic_regression_stats(self, notion_manager):
        """Get statistics for initial Logistic Regression model predictions only"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            'SELECT review_id FROM review_data WHERE model = "Logistic Regression Model"'
        )
        logistic_review_ids = [row['review_id'] for row in cursor.fetchall()]

        if not logistic_review_ids:
            return {
                "violation_count": 0,
                "category_counts": {},
                "page_counts": {}
            }

        # Collect category counts
        category_counts = {}

        # For each review, get violation counts by category
        review_ids_str = ','.join('?' for _ in logistic_review_ids)
        cursor.execute(
            f'''
            SELECT violation_type, COUNT(*) as count 
            FROM violation_data 
            WHERE review_id IN ({review_ids_str}) AND violation_type != 'compliant'
            GROUP BY violation_type
            ''',
            logistic_review_ids
        )

        for row in cursor.fetchall():
            category_counts[row['violation_type']] = row['count']

        # Count total violations
        total_violations = sum(category_counts.values())

        # Get pages with the most violations
        cursor.execute(
            f'''
            SELECT page_id, page_title, COUNT(*) as count 
            FROM violation_data 
            WHERE review_id IN ({review_ids_str}) AND violation_type != 'compliant'
            GROUP BY page_id, page_title
            ORDER BY count DESC
            ''',
            logistic_review_ids
        )

        page_counts = {}
        for row in cursor.fetchall():
            page_id = row['page_id']
            page_title = row['page_title']
            count = row['count']

            try:
                if page_id and page_id.strip():
                    url = notion_manager.get_page_url(page_id)
                    display_key = f"{page_title} ({url})"
                    page_counts[display_key] = count
                else:
                    page_counts[page_title] = count
            except Exception as e:
                print(f">>> Error fetching URL for page {page_id}: {e}")
                page_counts[page_title] = count

        conn.close()

        return {
            "violation_count": total_violations,
            "category_counts": category_counts,
            "page_counts": page_counts
        }

    def record_model_violations(self, review_id, notion_data, model_predictions):
        """Record initial model violations to violation_data table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for prediction in model_predictions:
                page_title = prediction.get("page_title", "")
                violations = prediction.get("violations", [])

                # Find the page_id from the URL
                page_id = None
                page_url = None
                for page in notion_data:
                    if page["title"] == page_title:
                        page_url = page["url"]
                        page_id = page["url"].split("-")[-1]
                        break

                # Save each violation type
                for violation_type in violations:
                    cursor.execute(
                        '''
                        INSERT INTO violation_data (review_id, page_id, page_title, page_url, violation_type, text, user_corrected)
                        VALUES (?, ?, ?, ?, ?, ?, 0)
                        ''',
                        (review_id, page_id, page_title, page_url, violation_type, prediction.get("text_to_review", ""))
                    )

            conn.commit()
            conn.close()
            print(f">>> Successfully recorded {len(model_predictions)} model predictions to database")
            return True

        except Exception as e:
            print(f">>> Error recording model violations to database: {e}")
            return False