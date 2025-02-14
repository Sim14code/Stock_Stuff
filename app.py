
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import joblib
import yfinance as yf
from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template


from model import* 

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

class SearchedStock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    price = db.Column(db.String(20), nullable=False)
    last_updated = db.Column(db.String(50), nullable=False)


# Load the model
MODEL_PATH = "stock_market_model.pkl"


try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    model = None  # Ensure model is set to None if loading fails


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    decision = None

    if request.method == 'POST':
        if not model:
            return render_template('predict.html', decision="Error: Model not loaded. Please check the server.")

        try:
            # Get form inputs
            open_price = float(request.form['open_price'])
            high_price = float(request.form['high_price'])
            low_price = float(request.form['low_price'])
            close_price = float(request.form['close_price'])
            volume = float(request.form['volume'])

            # Calculate additional feature
            price_change = (close_price - open_price) / open_price * 100
            features = np.array([[open_price, high_price, low_price, close_price, volume, price_change]])

            # Debug print
            print(f"🔍 Features: {features}")

            # Make prediction
            prediction = model.predict(features)[0]
            print(f"🧠 Prediction result: {prediction}")

            # Decision mapping
            decision = "Keep" if prediction == 1 else "Sell"
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            decision = f"Error in prediction: {e}"

    return render_template('predict.html', decision=decision)



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or Email already exists.', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your credentials.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    stocks = SearchedStock.query.filter_by(user_id=user_id).all()
    return render_template('index.html', stocks=stocks)

@app.route('/stock', methods=['GET', 'POST'])
def stock():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    searched_stocks = SearchedStock.query.filter_by(user_id=user_id).all()

    if request.method == 'POST':
        stock_symbol = request.form['symbol']
        stock_data = fetch_stock_data(stock_symbol)

        if 'error' not in stock_data:
            # Save searched stock for the logged-in user
            new_stock = SearchedStock(
                user_id=user_id,
                symbol=stock_data['symbol'],
                price=stock_data['price'],
                last_updated=stock_data['last_updated']
            )
            db.session.add(new_stock)
            db.session.commit()

        return render_template('stock.html', stock=stock_data, stocks=searched_stocks)

    return render_template('stock.html', stock=None, stocks=searched_stocks)

# Function to fetch stock data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.history(period='1d')  # Fetch data for today
        
        if stock_info.empty:
            return {'error': 'No data found for the given stock symbol. Please check the symbol.'}

        latest_price = stock_info['Close'].iloc[-1]  # Get the latest closing price
        last_updated = stock_info.index[-1].strftime('%Y-%m-%d %H:%M:%S')  # Get the last updated time

        return {
            'symbol': symbol,
            'price': f"${latest_price:.2f}",
            'last_updated': last_updated,
        }
    except Exception as e:
        return {'error': f'An unexpected error occurred: {e}'}


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
