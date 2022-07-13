from django.shortcuts import render,HttpResponse
import numpy as np
from io import BytesIO
import cv2
import numpy
# Create your views here.
def home(request):

    return render(request,'home.html') #render用来打开html
def login(request):
   if request.method=='GET':
        return render(request,'login.html')
   else:
     uname=request.POST.get('username')
     pwd = request.POST.get('password')
     if uname=='root' and pwd=='123':
        return render(request, 'home.html')
     else:
         return HttpResponse('用户名或密码错误！')

'''def login2(request):

    return render(request,'home.html')'''
def photo(request):
    file_object=request.FILES.get("pic")
    print(file_object.name)
    f=open('a1.jpg',mode='wb')
    for chunk in file_object.chunks():
        f.write(chunk)
    f.close()
    img = cv2.imread('D:/Mike/PycharmProjects/djangoProject/a1.jpg')
    #####Begin######
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    cv2.imwrite('D:/Mike/PycharmProjects/djangoProject/static/abc.jpg', b)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def index(request):
    '''img = cv2.imread('D:/Mike/PycharmProjects/djangoProject/static/dog0.jpg')
    #####Begin######
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    cv2.imwrite('D:/Mike/PycharmProjects/djangoProject/static/abc.jpg', b)
    cv2.imshow('b', b)'''
    return render(request,'index.html',{'image':'/a1.jpg'})
