from django.urls import re_path
from . import views

app_name = 'predict_1'

urlpatterns = [
    re_path(r'^(?P<pk>\d+)$', views.PredictRisk_1, name='predict_1'),
]
