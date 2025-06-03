from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from resume_parser import ResumeParser
from utils import extract_text_from_pdf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize parser
parser = ResumeParser()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return render_template("error.html", message="No file uploaded.")

    file = request.files['resume']
    if file.filename == '':
        return render_template("error.html", message="No file selected.")

    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)
        if text.startswith("Error:"):
            return render_template("error.html", message=text)

        # ✅ Extract job description from form
        job_description = request.form.get('job_description', '')

        # ✅ Analyze resume
        entities = parser.extract_entities(text)
        score_data = parser.score_resume(entities, job_description)

        os.remove(file_path)

        # ✅ Pass job description to results.html
        return render_template(
            'results.html',
            entities=entities,
            score_data=score_data,
            job_description=job_description
        )

    return render_template("error.html", message="Invalid file format. Please upload a PDF file.")

if __name__ == '__main__':
    app.run(debug=True)
