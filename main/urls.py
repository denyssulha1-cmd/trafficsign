from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('check_username/', views.check_username, name='check_username'),
    path('recognize/', views.recognize_sign, name='recognize'),
    path('about/', views.about, name='about'),
    path('history/', views.history_view, name='history'),
    path('feedback/', views.feedback_view, name='feedbacks'),
    path('feedback/edit/<int:feedback_id>/', views.edit_feedback, name='edit_feedback'),
    path('feedback/delete/<int:feedback_id>/', views.delete_feedback, name='delete_feedback'),
]
