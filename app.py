# # app.py
# from flask import Flask, render_template, request
# from user import user_bp
# from admin import admin_bp  # Import admin blueprint
# from models import db
# import os
# from dotenv import load_dotenv
# load_dotenv()
# from pyngrok import ngrok
# ngrok.set_auth_token('2j6yZx3KvVeSEOhGHG0bEjNdWB8_2ri6c36iDuZZxMJxxCD47')
# # app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# # postgresql://actiondetection_user:L0IBBidtXFP5ofA6wWxqqrOtNPIFa0yz@dpg-crh6li56l47c73c39fg0-a.oregon-postgres.render.com/actiondetection

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("database_url")
# app.config['SECRET_KEY'] = os.getenv("secret_key")  # Replace with your secret key


# # app = Flask(__name__)
# # app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("database_url")
# # app.config['SECRET_KEY'] = os.environ.get("secret_key")  # Replace with your secret key

# # Initialize the database
# db.init_app(app)
# def add_cache_control(response):
#     if request.endpoint in ['admin.admin_dashboard', 'admin.logout']:
#         response.headers['Cache-Control'] = 'no-store'
#         response.headers['Pragma'] = 'no-cache'
#         response.headers['Expires'] = '0'
#     return response
# # Register blueprints
# app.register_blueprint(user_bp)
# app.register_blueprint(admin_bp)  # Register admin blueprint

# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     # Ensure the database tables are created

#     with app.app_context():
#         db.create_all()
#     public_url = ngrok.connect(5000)
#     print("Public URL:", public_url)
#     app.run()
#     #app.run(debug=True)


# for hosting in render
from flask import Flask, render_template, request
from user import user_bp
from admin import admin_bp  # Import admin blueprint
from models import db
import os


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
    db.init_app(app)
    app.register_blueprint(user_bp)
    app.register_blueprint(admin_bp)

    with app.app_context():
        db.create_all()
    app.run()
    app.run(debug=True, port=8000)
