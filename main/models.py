from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings

from django.utils import timezone

class Feedbacks(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    name = models.CharField(max_length=100, verbose_name="Ім'я користувача")
    email = models.EmailField(verbose_name="Email")
    message = models.TextField(verbose_name="Відгук")
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.name} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Адміністратор'),
        ('user', 'Користувач'),
    )
    role = models.CharField(
        max_length=10,
        choices=ROLE_CHOICES,
        default='user',
        verbose_name='Роль користувача'
    )

    def __str__(self):
        return f"{self.username} ({self.role})"


class RecognitionHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='recognitions'
    )
    sign_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    image = models.ImageField(upload_to='recognized_signs/')
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sign_name} ({self.user.username})"


class Sign(models.Model):
    photo = models.ImageField(upload_to='signs_photos/', blank=True, null=True, verbose_name="Фото знака")
    name_sign = models.CharField(max_length=150, verbose_name="Назва знака")
    accuracy = models.DecimalField(max_digits=5, decimal_places=2, verbose_name="Точність (%)")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="Користувач")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата створення")

    def __str__(self):
        return f"{self.name_sign} ({self.accuracy}%)"
