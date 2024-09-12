# for hosting in render
from flask import Flask, render_template, request
from user import user
from admin import admin  # Import admin blueprint
from models import db
import os
# from dotenv import load_dotenv
# load_dotenv()


app = Flask(__name__)

# Initialize the database

def add_cache_control(response):
    if request.endpoint in ['admin.admin_dashboard', 'admin.logout']:
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response
# Register blueprints

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("database_url")
    app.config['SECRET_KEY'] = os.environ.get("secret_key")  # Replace with your secret key
    
    # app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("database_url")
    # app.config['SECRET_KEY'] = os.getenv("secret_key")
    
    db.init_app(app)
    app.register_blueprint(user)
    app.register_blueprint(admin)

    with app.app_context():
        db.create_all()
    app.run()
    # app.run(debug=True, port=8000)
