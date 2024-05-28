from django.shortcuts import render
from .forms import EmailForm
from .model import model, vectorizer

def classify_email(request):
    result = None
    if request.method == 'POST':
        form = EmailForm(request.POST)
        if form.is_valid():
            email_text = form.cleaned_data['email_text']
            email_vec = vectorizer.transform([email_text])
            prediction = model.predict(email_vec)[0]
            result = 'Spam' if prediction == 1 else 'Not Spam'
    else:
        form = EmailForm()

    return render(request, 'classify/email_form.html', {'form': form, 'result': result})
