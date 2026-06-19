from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Job Description
with open("job_description.txt", "r") as f:
    job = f.read()

# Resume Files
with open("resumes/resume1.txt", "r") as f:
    resume1 = f.read()

with open("resumes/resume2.txt", "r") as f:
    resume2 = f.read()

with open("resumes/resume3.txt", "r") as f:
    resume3 = f.read()

documents = [job, resume1, resume2, resume3]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(documents)

scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

print("Resume 1 Match Score:", round(scores[0] * 100, 2), "%")
print("Resume 2 Match Score:", round(scores[1] * 100, 2), "%")
print("Resume 3 Match Score:", round(scores[2] * 100, 2), "%")

best = scores.argmax() + 1
print("\nBest Resume: Resume", best)