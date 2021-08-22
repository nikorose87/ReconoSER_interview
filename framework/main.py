import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from facial_and_allign import face_alignment

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename1 = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		#See if it generates the photo
		img_CV(filename1)
		return render_template('upload.html', filename1=filename1)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_img(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

def img_CV(filename):
	#print('display_image filename: ' + filename)
	dir = os.path.join('static','uploads')
	CV = face_alignment(dir, filename)
	CV.perform_all()
	return 

if __name__ == "__main__":
    app.run()