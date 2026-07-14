from datetime import date, timedelta
import random
from datetime import date
from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Install faker first: pip install faker
from faker import Faker

DATABASE_URL = "sqlite:///company.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()


# ----------------------- #
# Customer
# ----------------------- #
class Customer(Base):
    __tablename__ = "customers"
    customer_id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)


# ----------------------- #
# Product
# ----------------------- #
class Product(Base):
    __tablename__ = "products"
    product_id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Float)
    stock = Column(Integer)


# ----------------------- #
# Orders
# ----------------------- #
class Order(Base):
    __tablename__ = "orders"
    order_id = Column(Integer, primary_key=True)
    customer_id = Column(
        Integer,
        ForeignKey("customers.customer_id"),
    )
    order_date = Column(Date)
    status = Column(String)
    total_amount = Column(Float)


# ----------------------- #
# Order Items
# ----------------------- #
class OrderItem(Base):
    __tablename__ = "order_items"
    item_id = Column(Integer, primary_key=True)
    order_id = Column(
        Integer,
        ForeignKey("orders.order_id"),
    )
    product_id = Column(
        Integer,
        ForeignKey("products.product_id"),
    )
    quantity = Column(Integer)
    price = Column(Float)


# ----------------------- #
# Shipment
# ----------------------- #
class Shipment(Base):
    __tablename__ = "shipments"
    shipment_id = Column(Integer, primary_key=True)
    order_id = Column(
        Integer,
        ForeignKey("orders.order_id"),
    )
    tracking_number = Column(String)
    carrier = Column(String)
    current_location = Column(String)
    estimated_delivery = Column(Date)
    status = Column(String)


# ----------------------- #
# Payment
# ----------------------- #
class Payment(Base):
    __tablename__ = "payments"
    payment_id = Column(Integer, primary_key=True)
    order_id = Column(
        Integer,
        ForeignKey("orders.order_id"),
    )
    payment_method = Column(String)
    payment_status = Column(String)
    amount = Column(Float)
    transaction_id = Column(String)


Base.metadata.create_all(engine)
db = Session()
fake = Faker()

# --- Configurable Data Size ---
NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 200
NUM_ORDERS = 5000

# 1. Insert Customers
print(f"Generating {NUM_CUSTOMERS} customers...")
customers = []
for _ in range(NUM_CUSTOMERS):
    customers.append(
        Customer(
            name=fake.name(), email=fake.unique.email(), phone=fake.phone_number()
        )
    )
db.add_all(customers)
db.commit()

# 2. Insert Products
print(f"Generating {NUM_PRODUCTS} products...")
products = []
product_pool = [
    "Laptop",
    "Smartphone",
    "Headphones",
    "Smartwatch",
    "Tablet",
    "Monitor",
    "Keyboard",
    "Mouse",
]
for _ in range(NUM_PRODUCTS):
    products.append(
        Product(
            name=f"{random.choice(product_pool)} {fake.word().capitalize()}",
            price=round(random.uniform(500.0, 120000.0), 2),
            stock=random.randint(5, 150),
        )
    )
db.add_all(products)
db.commit()

# 3. Generate Orders, Order Items, Shipments, and Payments
print(f"Generating {NUM_ORDERS} orders and related records...")
order_statuses = ["Pending", "Processing", "Shipped", "Delivered", "Cancelled"]
carriers = ["BlueDart", "Delhivery", "FedEx", "DHL"]
payment_methods = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"]
payment_statuses = ["Paid", "Pending", "Failed"]

for i in range(1, NUM_ORDERS + 1):
    # Order Date within the last year
    order_date = fake.date_between(start_date="-1y", end_date="today")
    status = random.choice(order_statuses)

    # Determine items for this order
    num_items = random.randint(1, 4)
    order_items_to_add = []
    total_amount = 0.0

    for _ in range(num_items):
        rand_prod_id = random.randint(1, NUM_PRODUCTS)
        qty = random.randint(1, 3)
        # Fetching price from our temporary list index to calculate total amount
        prod_price = products[rand_prod_id - 1].price
        item_total = prod_price * qty
        total_amount += item_total

        order_items_to_add.append(
            OrderItem(
                order_id=i, product_id=rand_prod_id, quantity=qty, price=prod_price
            )
        )

    # Create Order
    order = Order(
        customer_id=random.randint(1, NUM_CUSTOMERS),
        order_date=order_date,
        status=status,
        total_amount=round(total_amount, 2),
    )
    db.add(order)

    # Create Order Items
    db.add_all(order_items_to_add)

    # Create Shipment (if not pending or cancelled)
    if status in ["Shipped", "Delivered"]:
        shipment = Shipment(
            order_id=i,
            tracking_number=f"TRK{random.randint(100000, 999999)}",
            carrier=random.choice(carriers),
            current_location=f"{fake.city()} Hub",
            estimated_delivery=order_date + timedelta(days=random.randint(3, 7)),
            status="In Transit" if status == "Shipped" else "Delivered",
        )
        db.add(shipment)

    # Create Payment
    pay_status = "Paid" if status in ["Shipped", "Delivered"] else random.choice(payment_statuses)
    payment = Payment(
        order_id=i,
        payment_method=random.choice(payment_methods),
        payment_status=pay_status,
        amount=round(total_amount, 2),
        transaction_id=f"TXN{random.randint(100000, 999999)}",
    )
    db.add(payment)

    # Commit in batches of 500 to keep memory footprint low
    if i % 500 == 0:
        db.commit()
        print(f"Progress: {i}/{NUM_ORDERS} orders processed.")

db.commit()
print("Database Populated Successfully with Huge Dummy Data")
