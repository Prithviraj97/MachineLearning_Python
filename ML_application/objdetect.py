# #install required libraries to make simple object detector

# ERROR = "Try again."

# import math

# def computeHomeworkAvg(hw1, hw2, hw3):
#     return float(hw1 + hw2 + hw3) / 3

# def computeProjectAvg(proj1, proj2):
#     return float(proj1 + proj2) / 2

# def computeExamAvg(exam1, exam2):
#     return float(exam1 + exam2) / 2

# # Correct the weight calculation here
# def computeCourseAvg(hwavg, projavg, examavg):
#     return hwavg * 0.3 + projavg * 0.3 + examavg * 0.4

# def avgToLetterGrade(courseavg):
#     if 100 >= courseavg >= 90:
#         return 'A'
#     elif 90 > courseavg >= 80:
#         return 'B'
#     elif 80 > courseavg >= 70:
#         return 'C'
#     elif 70 > courseavg >= 60:
#         return 'D'
#     elif 60 > courseavg:
#         return 'F'

# # Pass necessary variables as arguments
# def printGradeReport(hwavg, projavg, examavg, courseavg, LetterGrade):
#     print(" Homework average (30% of grade): {:0.2f}".format(hwavg))
#     print(" Project average (30% of grade): {:0.2f}".format(projavg))
#     print(" Exam average (40% of grade): {:0.2f}".format(examavg))
#     print(" Student course average: {:0.2f}".format(courseavg))
#     print(" Course grade: " + LetterGrade)

# while True:
#     name = input("Enter the student's name (or 'stop'): ")
#     if name.strip().lower() != 'stop':
#         print()
#         print("HOMEWORK:")
#         hw1, hw2, hw3 = [float(input(f" Enter HW{i} grade: ")) for i in range(1, 4)]
#         print()
#         print("PROJECTS:")
#         proj1, proj2 = [float(input(f" Enter Pr{i} grade: ")) for i in range(1, 3)]
#         print()
#         print("EXAM:")
#         exam1, exam2 = [float(input(f" Enter Ex{i} grade: ")) for i in range(1, 3)]
        
#         print()
#         print("Grade report for: " + name)
#         hwavg = computeHomeworkAvg(hw1, hw2, hw3)
#         projavg = computeProjectAvg(proj1, proj2)
#         examavg = computeExamAvg(exam1, exam2)
#         courseavg = computeCourseAvg(hwavg, projavg, examavg)
#         LetterGrade = avgToLetterGrade(courseavg)
        
#         # Call printGradeReport with the necessary variables
#         printGradeReport(hwavg, projavg, examavg, courseavg, LetterGrade)
#         print()
#     else:
#         print("Thanks for using the grade calculator! Goodbye.")
#         break

def compute_avg(values):
    return sum(values) / len(values)

def compute_course_avg(hw_avg, proj_avg, exam_avg):
    return hw_avg * 0.3 + proj_avg * 0.3 + exam_avg * 0.4

def avg_to_letter_grade(course_avg):
    if course_avg >= 90:
        return 'A'
    elif course_avg >= 80:
        return 'B'
    elif course_avg >= 70:
        return 'C'
    elif course_avg >= 60:
        return 'D'
    else:
        return 'F'

def print_grade_report(name, hw_avg, proj_avg, exam_avg, course_avg, letter_grade):
    print(f"Grade report for: {name}")
    print("Homework average (30% of grade): {:.2f}".format(hw_avg))
    print("Project average (30% of grade): {:.2f}".format(proj_avg))
    print("Exam average (40% of grade): {:.2f}".format(exam_avg))
    print("Student course average: {:.2f}".format(course_avg))
    print("Course grade:", letter_grade)
    print()

while True:
    name = input("Enter the student's name (or 'stop'): ").strip()
    if name.lower() == 'stop':
        print("Thanks for using the grade calculator! Goodbye.")
        break
    
    print(f"Grade report for: {name}")
    
    hw_grades = [float(input(f"Enter HW{i} grade: ")) for i in range(1, 4)]
    proj_grades = [float(input(f"Enter Pr{i} grade: ")) for i in range(1, 3)]
    exam_grades = [float(input(f"Enter Ex{i} grade: ")) for i in range(1, 3)]
    
    hw_avg = compute_avg(hw_grades)
    proj_avg = compute_avg(proj_grades)
    exam_avg = compute_avg(exam_grades)
    course_avg = compute_course_avg(hw_avg, proj_avg, exam_avg)
    letter_grade = avg_to_letter_grade(course_avg)
    
    print_grade_report(name, hw_avg, proj_avg, exam_avg, course_avg, letter_grade)
