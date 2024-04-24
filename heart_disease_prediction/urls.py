from django.contrib import admin
from django.urls import path, include
from django.urls import re_path

from accounts import views
from heart_disease_prediction import settings

from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^$', views.user_login, name='user_login'),
    path('accounts/', include('accounts.urls')),
    path('predict/', include('predict_risk.urls')),
    path('predict_1/', include('predict_risk_1.urls')),
]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
