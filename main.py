from fastapi import FastAPI, Form
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Set
from uuid import UUID
from datetime import date, datetime, time, timedelta
app = FastAPI()

class Event(BaseModel):
    event_id : UUID
    start_date: date
    start_time: datetime
    end_time : datetime
    repeat_time: time
    execute_after: timedelta


class Profile(BaseModel):
    name: str
    email: str
    age: int

class Image(BaseModel):
    url:HttpUrl
    name:str

class Product(BaseModel):
    name:str = Field(example="phone")
    price:int = Field(title="Price of the item",description="price of the item added")
    discount:int
    discounted_price:int
    tags:  List[str]= Field(example="[electronics,phones]")
    image:List[Image]

    class Config:
        schema_extra = {
            "example": {
                "name": "Phone",
                "price": 100,
                "discount": 10,
                "discounted_price": 90,
                "tags": ["electronics", "computers"],
                "image": [
                    {"url": "http://www.google.com", "name": "phone image"},
                    {"url": "http://www.google.com", "name": "phone image side view"}
                ]
            }
        }


class Offer(BaseModel):
    name:str
    description:str
    price:float
    products: List[Product]


class User(BaseModel):
    name:str
    email:str




@app.post('/addevent')
def addevent(event:Event):
    return event

@app.post('/addoffer')
def addoffer(offer:Offer):
    return offer



@app.post('/purchase')
def purchase(user:User,product:Product):
    return {'user':user,"product":product}


@app.post('/addproduct/{product_id}')
def addproduct(product:Product,product_id:int, category:str):
    product.discounted_price= product.price-(product.price*product.discount)/100
    return {"product_id":product_id,"product":product,'category':category}


@app.get('/')
def index():
    return 'Hello world!'
@app.get('/property/{id:int}')
def property(id):
    return {f'This is a property page for property {id}'}

@app.get('/profile/{username}')
def profile(username):
    return {f'This a profile page for a user: {username}'}

@app.get('/movies')
def movie_list():
    return {'movie list':{'avatar no.1','shaktiman no.1'}}

@app.get('/user/{username}')
def profile(username):
    return {f'The is a profile page for {username}'}


@app.get('/user/admin')
def admin():
    return {'This is a admin page'}

@app.get('/products')
def products(id:int=1,price:int=0):
    return {f'Product with an id:{id} and {price}'}

@app.get('/profile/{userid}/comments')
def profile(userid:int,commentid:int):
    return {f'Profile page for user with user id {userid} and comment with {commentid}'}

@app.post('/adduser')
def adduser(profile:Profile,age:int):
    return profile
