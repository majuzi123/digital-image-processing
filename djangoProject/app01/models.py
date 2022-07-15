from django.db import models

# Create your models here.
pic = models.ImageField(upload_to='pic/',verbose_name=u'图片地址')