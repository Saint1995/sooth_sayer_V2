from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Doc_Summarizer(models.Model):
    user_id = models.ForeignKey(User, on_delete=models.CASCADE,)
    web_link = models.CharField(max_length=100)

    class Meta:
        verbose_name_plural = 'summarizer user'

    def __unicode__(self):
        return u"%s's Web Summarize User Info" % self.user_id