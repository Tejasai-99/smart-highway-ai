from django.db import models

class Analysis(models.Model):

    video_a = models.FileField(upload_to='videos/')

    video_b = models.FileField(upload_to='videos/')

    total_vehicles = models.IntegerField(default=0)

    safe_vehicles = models.IntegerField(default=0)

    missing_vehicles = models.IntegerField(default=0)

    cars = models.IntegerField(default=0)

    trucks = models.IntegerField(default=0)

    bikes = models.IntegerField(default=0)

    buses = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):

        return f"Analysis {self.id}"