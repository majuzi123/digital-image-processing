"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from app01 import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/', views.home),
    path('login/', views.login),
    path('photo/', views.photo),
    path('index/', views.index),
    path('f1_b/', views.f1_b),
    path('f1_g/', views.f1_g),
    path('f1_r/', views.f1_r),
    path('f2_h/', views.f2_h),
    path('f2_s/', views.f2_s),
    path('f2_v/', views.f2_v),
    path('f3/', views.f3),
    path('f4_1/', views.f4_1),
    path('f4_2/', views.f4_2),
    path('f4_3/', views.f4_3),
    path('f5/', views.f5),
    path('f6/', views.f6),
    path('f7/', views.f7),
    path('f8/', views.f8),
    path('f9/', views.f9),
    path('f10/', views.f10),
    path('f11/', views.f11),
    path('f12/', views.f12),
    path('f13_op/', views.f13_op),
    path('f13_cl/', views.f13_cl),
    path('f14/', views.f14),
    path('f15/', views.f15),
    #path('login2/', views.login2),
]
