from flask import Flask, render_template, request, redirect, flash, session
import pandas as pd
import numpy as np

from flask import request, redirect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

app = Flask(__name__)
app.secret_key = "salesanalytics"


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/dashboard')
def dashboard():

    # ==========================
    # Load Dataset
    # ==========================

    df = pd.read_excel("uploads/sales_data.xlsx")

    df['SalesDate'] = pd.to_datetime(df['SalesDate'])

    df['Revenue'] = (
        df['Quantity'] *
        df['UnitPrice']
    )

    # ==========================
    # KPI Metrics
    # ==========================

    total_revenue = round(df['Revenue'].sum(), 2)

    total_orders = len(df)

    top_product = (
        df.groupby('Product')['Revenue']
        .sum()
        .idxmax()
    )

    best_region = (
        df.groupby('Region')['Revenue']
        .sum()
        .idxmax()
    )

    # ==========================
    # Region Chart
    # ==========================

    region_sales = (
        df.groupby('Region')['Revenue']
        .sum()
        .reset_index()
    )

    region_labels = region_sales['Region'].tolist()
    region_values = region_sales['Revenue'].tolist()

    # ==========================
    # Product Chart
    # ==========================

    product_sales = (
        df.groupby('Product')['Revenue']
        .sum()
        .reset_index()
    )

    product_labels = product_sales['Product'].tolist()
    product_values = product_sales['Revenue'].tolist()

    # ==========================
    # Monthly Trend
    # ==========================

    df['Month'] = df['SalesDate'].dt.strftime('%b')

    monthly_sales = (
        df.groupby('Month')['Revenue']
        .sum()
        .reset_index()
    )

    month_labels = monthly_sales['Month'].tolist()
    month_values = monthly_sales['Revenue'].tolist()

    # ==========================
    # Machine Learning
    # ==========================

    ml_df = df.copy()

    le_product = LabelEncoder()
    le_category = LabelEncoder()
    le_region = LabelEncoder()

    ml_df['Product'] = le_product.fit_transform(
        ml_df['Product']
    )

    ml_df['Category'] = le_category.fit_transform(
        ml_df['Category']
    )

    ml_df['Region'] = le_region.fit_transform(
        ml_df['Region']
    )

    X = ml_df[
        [
            'Product',
            'Category',
            'Quantity',
            'UnitPrice',
            'Region'
        ]
    ]

    y = ml_df['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = round(
        mean_absolute_error(
            y_test,
            y_pred
        ),
        2
    )

    rmse = round(
        np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        ),
        2
    )

    r2 = round(
        r2_score(
            y_test,
            y_pred
        ),
        4
    )

    # ==========================
    # Dashboard Page
    # ==========================

    return render_template(
        "dashboard.html",

        total_revenue=total_revenue,
        total_orders=total_orders,
        top_product=top_product,
        best_region=best_region,

        region_labels=region_labels,
        region_values=region_values,

        product_labels=product_labels,
        product_values=product_values,

        month_labels=month_labels,
        month_values=month_values,

        mae=mae,
        rmse=rmse,
        r2=r2
    )


@app.route('/prediction')
def prediction():

    df = pd.read_excel("uploads/sales_data.xlsx")

    df['SalesDate'] = pd.to_datetime(df['SalesDate'])

    df['Revenue'] = (
        df['Quantity'] *
        df['UnitPrice']
    )

    ml_df = df.copy()

    le_product = LabelEncoder()
    le_category = LabelEncoder()
    le_region = LabelEncoder()

    ml_df['Product'] = le_product.fit_transform(
        ml_df['Product']
    )

    ml_df['Category'] = le_category.fit_transform(
        ml_df['Category']
    )

    ml_df['Region'] = le_region.fit_transform(
        ml_df['Region']
    )

    X = ml_df[
        [
            'Product',
            'Category',
            'Quantity',
            'UnitPrice',
            'Region'
        ]
    ]

    y = ml_df['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = round(
        mean_absolute_error(
            y_test,
            y_pred
        ),
        2
    )

    rmse = round(
        np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        ),
        2
    )

    r2 = round(
        r2_score(
            y_test,
            y_pred
        ),
        4
    )

    return render_template(
        'prediction.html',
        mae=mae,
        rmse=rmse,
        r2=r2
    )

@app.route('/revenue')
def revenue():

    df = pd.read_excel("uploads/sales_data.xlsx")

    df['SalesDate'] = pd.to_datetime(df['SalesDate'])

    df['Revenue'] = (
        df['Quantity'] *
        df['UnitPrice']
    )

    total_revenue = round(
        df['Revenue'].sum(),
        2
    )

    average_revenue = round(
        df['Revenue'].mean(),
        2
    )

    df['Month'] = (
        df['SalesDate']
        .dt.strftime('%b')
    )

    monthly_sales = (
        df.groupby('Month')['Revenue']
        .sum()
        .reset_index()
    )

    month_labels = (
        monthly_sales['Month']
        .tolist()
    )

    month_values = (
        monthly_sales['Revenue']
        .tolist()
    )

    best_month = (
        monthly_sales.loc[
            monthly_sales['Revenue'].idxmax(),
            'Month'
        ]
    )

    return render_template(
        'revenue.html',
        total_revenue=total_revenue,
        average_revenue=average_revenue,
        best_month=best_month,
        month_labels=month_labels,
        month_values=month_values
    )

@app.route('/product')
def product():

    df = pd.read_excel("uploads/sales_data.xlsx")

    df['Revenue'] = (
        df['Quantity'] *
        df['UnitPrice']
    )

    product_sales = (
        df.groupby('Product')['Revenue']
        .sum()
        .reset_index()
        .sort_values(
            by='Revenue',
            ascending=False
        )
    )

    top_product = (
        product_sales.iloc[0]['Product']
    )

    top_product_revenue = round(
        product_sales.iloc[0]['Revenue'],
        2
    )

    total_products = (
        product_sales['Product']
        .nunique()
    )

    product_labels = (
        product_sales['Product']
        .tolist()
    )

    product_values = (
        product_sales['Revenue']
        .tolist()
    )

    product_table = (
        product_sales
        .to_dict('records')
    )

    return render_template(
        'product.html',

        top_product=top_product,

        top_product_revenue=top_product_revenue,

        total_products=total_products,

        product_labels=product_labels,

        product_values=product_values,

        product_table=product_table
    )

@app.route('/region')
def region():

    df = pd.read_excel("uploads/sales_data.xlsx")

    df['Revenue'] = (
        df['Quantity'] *
        df['UnitPrice']
    )

    region_sales = (
        df.groupby('Region')['Revenue']
        .sum()
        .reset_index()
        .sort_values(
            by='Revenue',
            ascending=False
        )
    )

    best_region = (
        region_sales.iloc[0]['Region']
    )

    best_region_revenue = round(
        region_sales.iloc[0]['Revenue'],
        2
    )

    total_regions = (
        region_sales['Region']
        .nunique()
    )

    region_labels = (
        region_sales['Region']
        .tolist()
    )

    region_values = (
        region_sales['Revenue']
        .tolist()
    )

    return render_template(
        'region.html',

        best_region=best_region,

        best_region_revenue=best_region_revenue,

        total_regions=total_regions,

        region_labels=region_labels,

        region_values=region_values
    )

@app.route('/reports')
def reports():

    df = pd.read_excel("uploads/sales_data.xlsx")

    df['SalesDate'] = pd.to_datetime(df['SalesDate'])

    df['Revenue'] = (
        df['Quantity'] *
        df['UnitPrice']
    )

    total_revenue = round(
        df['Revenue'].sum(),
        2
    )

    total_orders = len(df)

    top_product = (
        df.groupby('Product')['Revenue']
        .sum()
        .idxmax()
    )

    best_region = (
        df.groupby('Region')['Revenue']
        .sum()
        .idxmax()
    )

    # Machine Learning

    ml_df = df.copy()

    le_product = LabelEncoder()
    le_category = LabelEncoder()
    le_region = LabelEncoder()

    ml_df['Product'] = le_product.fit_transform(
        ml_df['Product']
    )

    ml_df['Category'] = le_category.fit_transform(
        ml_df['Category']
    )

    ml_df['Region'] = le_region.fit_transform(
        ml_df['Region']
    )

    X = ml_df[
        [
            'Product',
            'Category',
            'Quantity',
            'UnitPrice',
            'Region'
        ]
    ]

    y = ml_df['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = round(
        mean_absolute_error(
            y_test,
            y_pred
        ),
        2
    )

    rmse = round(
        np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        ),
        2
    )

    r2 = round(
        r2_score(
            y_test,
            y_pred
        ),
        4
    )

    return render_template(
        'reports.html',

        total_revenue=total_revenue,
        total_orders=total_orders,

        top_product=top_product,
        best_region=best_region,

        mae=mae,
        rmse=rmse,
        r2=r2
    )
    
@app.route('/forecast')
def forecast():

    df = pd.read_excel("uploads/sales_data.xlsx")

    df['SalesDate'] = pd.to_datetime(df['SalesDate'])

    df['Revenue'] = (
        df['Quantity'] *
        df['UnitPrice']
    )

    df['MonthNum'] = (
        df['SalesDate']
        .dt.month
    )

    monthly_sales = (
        df.groupby('MonthNum')['Revenue']
        .sum()
        .reset_index()
    )

    X = monthly_sales[['MonthNum']]
    y = monthly_sales['Revenue']

    model = LinearRegression()

    model.fit(X, y)

    future_month = (
        monthly_sales['MonthNum']
        .max() + 1
    )

    predicted_revenue = round(
        model.predict([[future_month]])[0],
        2
    )

    month_labels = (
        monthly_sales['MonthNum']
        .astype(str)
        .tolist()
    )

    month_values = (
        monthly_sales['Revenue']
        .tolist()
    )

    return render_template(
        "forecast.html",

        future_month=future_month,

        predicted_revenue=predicted_revenue,

        month_labels=month_labels,

        month_values=month_values
    )
    
from flask import send_file

@app.route('/download-report')
def download_report():

    df = pd.read_excel("uploads/sales_data.xlsx")

    df['Revenue'] = (
        df['Quantity'] *
        df['UnitPrice']
    )

    total_revenue = round(
        df['Revenue'].sum(),
        2
    )

    total_orders = len(df)

    top_product = (
        df.groupby('Product')['Revenue']
        .sum()
        .idxmax()
    )

    best_region = (
        df.groupby('Region')['Revenue']
        .sum()
        .idxmax()
    )

    pdf_path = "reports/Sales_Report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "Sales Analytics Report",
            styles['Title']
        )
    )

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            f"Total Revenue: ₹{total_revenue}",
            styles['BodyText']
        )
    )

    content.append(
        Paragraph(
            f"Total Orders: {total_orders}",
            styles['BodyText']
        )
    )

    content.append(
        Paragraph(
            f"Top Product: {top_product}",
            styles['BodyText']
        )
    )

    content.append(
        Paragraph(
            f"Best Region: {best_region}",
            styles['BodyText']
        )
    )

    doc.build(content)

    return send_file(
        pdf_path,
        as_attachment=True
    )
    
@app.route('/upload', methods=['GET','POST'])
def upload():

    if request.method == 'POST':

        file = request.files['file']

        if file:

            file.save(
                "uploads/sales_data.xlsx"
            )
            
            flash("Dataset uploaded successfully!")

            return redirect(
                "/upload"
            )

    return render_template(
        'upload.html'
    )
    
@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin123":

            session['user'] = username

            return redirect('/dashboard')

        else:

            return render_template(
                'login.html',
                error="Invalid Username or Password"
            )

    return render_template('login.html')

@app.route('/logout')
def logout():

    session.clear()

    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)