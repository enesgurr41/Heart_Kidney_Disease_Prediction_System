from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class UserProfileInfo(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    profile_pic = models.ImageField(upload_to='profile_pics', default="1.jpg")

    def __str__(self):
        return self.user.username

    def get_absolute_url(self):
        return reverse('profile', args=[str(self.user.id)])
