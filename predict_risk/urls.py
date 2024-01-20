from django.urls import re_path
from . import views

app_name = 'predict'

urlpatterns = [
    re_path(r'^(?P<pk>\d+)$', views.PredictRisk, name='predict'),
]
