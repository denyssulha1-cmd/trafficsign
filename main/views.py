from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Sign
from django.utils import timezone
from .ml_model.predict import predict_sign
import os
from django.conf import settings
from .models import RecognitionHistory

from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from .models import Feedbacks

User = get_user_model()

        
def home(request):
    return render(request, 'main/home.html')

def about(request):
    return render(request, 'main/about.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f'Вітаємо, {username}!')
            return redirect('home')
        else:
            messages.error(request, 'Невірний логін або пароль!')

    return render(request, 'main/login.html')


def register_view(request):
    if request.method == 'POST':
        surname = request.POST.get('surname')
        nameuser = request.POST.get('nameuser')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        role = 'user'

        # Перевірка унікальності логіну
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Такий логін вже існує!')
            return redirect('register')

        # Перевірка паролів
        if password1 != password2:
            messages.error(request, 'Паролі не співпадають!')
            return redirect('register')

        # Створення користувача
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
            first_name=nameuser,
            last_name=surname,
            role=role
        )
        user.save()

        messages.success(request, 'Реєстрація успішна! Тепер увійдіть.')
        return redirect('login')

    return render(request, 'main/register.html')



def check_username(request):
    username = request.GET.get('username', None)
    exists = User.objects.filter(username=username).exists()
    return JsonResponse({'exists': exists})


def logout_view(request):
    logout(request)
    messages.info(request, 'Ви вийшли з акаунта.')
    return redirect('home')

@login_required(login_url='login')
def recognize_sign(request):
    if request.method == "POST" and request.FILES.get("photo"):
        photo = request.FILES["photo"]

        # Шлях, куди тимчасово збережеться зображення
        photo_path = os.path.join(settings.MEDIA_ROOT, photo.name)

        # Зберігаємо файл
        with open(photo_path, "wb+") as destination:
            for chunk in photo.chunks():
                destination.write(chunk)

        # 👇 Викликаємо саме ML-функцію
        name_sign, accuracy = predict_sign(photo_path)

        # Зберігаємо результат у базі
        Sign.objects.create(
            photo=photo.name,
            name_sign=name_sign,
            accuracy=accuracy,
            user=request.user
        )
        if request.user.is_authenticated:
            RecognitionHistory.objects.create(
                user=request.user,
                sign_name=name_sign,  
                image=photo.name,
                accuracy=accuracy
            )
        return render(request, "main/recognize.html", {
            "success": True,
            "name_sign": name_sign,
            "accuracy": accuracy
        })

    return render(request, "main/recognize.html")

@login_required
def history_view(request):
    history = RecognitionHistory.objects.filter(user=request.user).order_by('-date')
    return render(request, 'main/history.html', {'history': history})

def feedback_view(request):
    feedbacks = Feedbacks.objects.order_by('-created_at')
    
    if request.method == 'POST':
        # Тільки авторизовані користувачі можуть додавати відгуки
        if request.user.is_authenticated:
            message = request.POST.get('message')
            if not message.strip():
                messages.error(request, "Відгук не може бути порожнім.")
                return redirect('feedbacks')

            Feedbacks.objects.create(
                user=request.user,
                name=request.user.get_full_name() or request.user.username,
                email=request.user.email,
                message=message
            )
            messages.success(request, "Відгук успішно додано!")
            return redirect('feedbacks')
        else:
            messages.error(request, "Щоб залишити відгук, увійдіть у акаунт.")
            return redirect('login')

    context = {'feedbacks': feedbacks}
    return render(request, 'main/feedbacks.html', context)


@login_required
def edit_feedback(request, feedback_id):
    feedback = get_object_or_404(Feedbacks, id=feedback_id)

    # Автор може редагувати тільки свої коментарі
    if request.user != feedback.user:
        messages.error(request, "Ви можете редагувати тільки свої відгуки.")
        return redirect('feedbacks')

    if request.method == 'POST':
        message = request.POST.get('message')
        if not message.strip():
            messages.error(request, "Відгук не може бути порожнім.")
            return redirect('edit_feedback', feedback_id=feedback.id)

        feedback.message = message
        feedback.save()
        messages.success(request, "Відгук успішно оновлено!")
        return redirect('feedbacks')

    context = {'feedback': feedback}
    return render(request, 'main/edit_feedback.html', context)


@login_required
def delete_feedback(request, feedback_id):
    feedback = get_object_or_404(Feedbacks, id=feedback_id)

    # Автор може видаляти тільки свої відгуки
    if request.user.role != 'admin' and request.user != feedback.user:
        messages.error(request, "Ви можете видаляти тільки свої відгуки.")
        return redirect('feedbacks')

    feedback.delete()
    messages.success(request, "Відгук видалено!")
    return redirect('feedbacks')
