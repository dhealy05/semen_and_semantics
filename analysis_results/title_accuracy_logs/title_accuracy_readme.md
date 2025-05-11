# Title Accuracy

Run "run_title_accuracy.py" to get a random sample of videos with metadata saved to a JSON. The "title_accuracy" field will be null by default.

Titles can be drawn from the total collection, from years, or by category. To find reasonable category samples, I found N nearest neighbors from a reference embedding (i.e. "incest") and then took a random sample of size M; I wanted some reasonably nearest neighbors, but not necessarily the very top, in case they were true outliers.

For the sample chosen, you will see a list of titles like so:

Metadata(title='Cum Over and Fuck Me Missy Stone', url='/view_video.php?viewkey=832515857', views='1588526', duration='34:45', timestamp=datetime.datetime(2015, 11, 20, 0, 0))

In order to gauge title accuracy, a reviewer must navigate to the URL and weight the video contents against the video title. We use a scale of 1-5:

1: Completely inaccurate; title is not relevant to video
2: Ambiguous leaning inaccurate
3: Ambiguous without a clear lean towards fact or fiction
4: Ambiguous leaning accurate
5: Complete accurate; title is clearly describing video contents

I have included my own review scores and accompanying metadata in the various *_title_accuracy.json log files. Many links are dead and yield some result like "video removed"; these remain "null" indicating they have not been scored.

Generally I find that the available titles reflect the contents. An exception would be index 3 of incest_title_accuracy.json, entitled "Cum in panties step sister", but which did not depict any incestuous content; in that case the "step sister" suffix was pure SEO.

Playlists, which are collections of videos, I estimated based on the total contents.

After review, you can run the "analyze_title_accuracy" function to see the total scores. Here's a sample for some of the provided files:

File: ../analysis_results/title_accuracy_logs/title_accuracy_2014.json
Total samples: 10
Video Not Available (Null): 7/10 Samples
Average Score (for available videos): 5.00

File: ../analysis_results/title_accuracy_logs/incest_title_accuracy.json
Total samples: 10
Video Not Available (Null): 4/10 Samples
Average Score (for available videos): 3.67

File: ../analysis_results/title_accuracy_logs/overall_title_accuracy_0.json
Total samples: 10
Video Not Available (Null): 8/10 Samples
Average Score (for available videos): 5.00

Overall Statistics:
Total samples across all files: 30
Video Not Available (Null): 19/30 Samples
Average Score (for available videos): 4.27
