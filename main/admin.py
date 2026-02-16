from django.contrib import admin
from .models import Sign

@admin.register(Sign)
class SignAdmin(admin.ModelAdmin):
    list_display = ('id', 'name_sign', 'accuracy', 'user', 'created_at')
    list_filter = ('created_at', 'user')
    search_fields = ('name_sign',)

