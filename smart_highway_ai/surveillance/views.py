from django.shortcuts import render
from .models import Analysis
from .detector import run_detection

def home(request):

    message = ""

    stats = None

    if request.method == "POST":

        video_a = request.FILES.get("video_a")

        video_b = request.FILES.get("video_b")

        if video_a and video_b:

            # Save uploaded videos
            analysis = Analysis.objects.create(
                video_a=video_a,
                video_b=video_b
            )

            # Run detection on Camera A
            result_a = run_detection(
                analysis.video_a.path
            )

            # Run detection on Camera B
            result_b = run_detection(
                analysis.video_b.path
            )

            # Combine totals
            total = (
                result_a["total"] +
                result_b["total"]
            )

            cars = (
                result_a["cars"] +
                result_b["cars"]
            )

            trucks = (
                result_a["trucks"] +
                result_b["trucks"]
            )

            bikes = (
                result_a["bikes"] +
                result_b["bikes"]
            )

            buses = (
                result_a["buses"] +
                result_b["buses"]
            )

            # Demo SAFE/MISSING logic
            safe = int(total * 0.8)

            missing = total - safe

            # =========================
            # ACCURACY CALCULATION
            # =========================

            # Example manual actual count
            actual_vehicles = total + 5

            accuracy = (
                total / actual_vehicles
            ) * 100

            accuracy = round(accuracy, 2)

            # =========================
            # SAVE DATABASE
            # =========================

            analysis.total_vehicles = total

            analysis.safe_vehicles = safe

            analysis.missing_vehicles = missing

            analysis.cars = cars

            analysis.trucks = trucks

            analysis.bikes = bikes

            analysis.buses = buses

            analysis.save()

            # =========================
            # SEND TO FRONTEND
            # =========================

            stats = {
                "total": total,
                "safe": safe,
                "missing": missing,
                "cars": cars,
                "trucks": trucks,
                "bikes": bikes,
                "buses": buses,
                "accuracy": accuracy
            }

            message = "AI Analysis Completed!"

    return render(
        request,
        "home.html",
        {
            "message": message,
            "stats": stats
        }
    )