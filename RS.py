# Import the Streamlit library and alias it as 'st'
import streamlit as st

# Load pre-trained machine learning model from saved files
import pickle

# Text pattern matching using regular expressions
import re

# Text Processing and working with human language data
import nltk 

# Tokenization models and 'stopword' dataset from NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Loading models and opening the files in binary read mode
clf = pickle.load(open('clf (2).pkl', 'rb'))
tfidf = pickle.load(open('tfidf (2).pkl', 'rb'))

# Clean resume text
def clean_resume(resume_text):
    # Remove URLs (links) from the text
    cleanText = re.sub('http\S+\s', ' ', resume_text )
    # Remove 'RT' and 'cc'
    cleanText = re.sub('RT|cc', ' ', cleanText)
    # Remove hashtags
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    # Remove mentions (username)
    cleanText = re.sub('@\S+', '  ', cleanText)
    # Remove special characters
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    # Remove non-ASCII characters
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    # Remove extra whitespace and replace it with a single space
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Web App
def main():
    # Set the title
     st.title("Resume Screening App")
    # To upload resumes in either 'txt' or 'pdf' format
     uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

     # Check if an uploaded file exists and is not None
     if uploaded_file is not None:
        try:
            # Read the content of the uploaded file
            resume_bytes = uploaded_file.read()
            # Decode the bytes as UTF-8 to obtain the resume text
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

       # Mapping of 'prediction_id' with 'category_name
        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display text, data, or visualizations in app
        st.write("Predicted Category:", category_name)



# python main
if __name__ == "_main_": main()
