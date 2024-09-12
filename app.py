# for hosting in render
from flask import Flask, render_template, request
from user import user_bp
from admin import admin_bp  # Import admin blueprint
from models import db
import os
# from dotenv import load_dotenv
# load_dotenv()


app = Flask(__name__)

# for development
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("database_url")
app.config['SECRET_KEY'] = os.environ.get("secret_key")  # Replace with your secret key

# for local
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("database_url")
# app.config['SECRET_KEY'] = os.getenv("secret_key")

# Initialize the database
db.init_app(app)

def add_cache_control(response):
    if request.endpoint in ['admin_bp.admin_dashboard', 'admin_bp.logout']:
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response
# Register blueprints

app.register_blueprint(user_bp)
app.register_blueprint(admin_bp)


print("0")

@app.route('/')
def index():
    return render_template('index.html')

print("1")
if __name__ == '__main__':
    print("2")

    with app.app_context():
        db.create_all()
    
    # app.run()
    print("3")
    port = int(os.environ.get('PORT', 5000)) 
    print("4")
    app.run(host='0.0.0.0', port=port)
