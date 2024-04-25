from django.urls import path, re_path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('register/', views.register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    path('recipe/', views.recipe, name='recipe'),
    path('single/', views.single, name='single'),
    path('single1/', views.single1, name='single1'),
    path('profile/<int:pk>/', views.ProfileDetailView.as_view(), name='profile'),
    # path('profile/<int:pk>/edit/', views.profile_update, name='edit_profile'),
]
