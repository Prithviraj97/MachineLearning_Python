import xlsxwriter
# import string

# Create a workbook object
workbook = xlsxwriter.Workbook('C:\\Users\\TheEarthG\\source\\repos\\MachineLearning_Python\\test2.xlsx')

# def write_column_headers(worksheet, headers):
#     """
#     Write column headers to the first row of the worksheet.

#     Args:
#         worksheet: The worksheet object to write the headers to.
#         headers (list): A list of column header strings.
#     """
#     max_columns = len(string.ascii_uppercase)
#     for col, header in enumerate(headers, start=1):
#         if col > max_columns:
#             raise ValueError("Too many columns for the alphabet range.")
#         column_letter = string.ascii_uppercase[col - 1]
#         worksheet.write(f"{column_letter}1", header)

# # Example usage
# worksheet = workbook.add_worksheet()
# column_headers = [
#     'total site energy', "heating", "cooling", "fans", "pumps", " Heating Gas",
#     "elec fac(j)_jan", "elec fac(j)_feb", "elec fac(j)_mar", "elec fac(j)_apr",
#     "elec fac(j)_may", "elec fac(j)_jun", "elec fac(j)_jul", "elec fac(j)_aug",
#     "elec fac(j)_sep", "elec fac(j)_oct", "elec fac(j)_nov", "elec fac(j)_dec",
#     "gas fac(j)_jan", "gas fac(j)_feb", "gas fac(j)_mar", "gas fac(j)_apr",
#     "gas fac(j)_may", "gas fac(j)_jun", "gas fac(j)_jul", "gas fac(j)_aug",
#     "gas fac(j)_sep", "gas fac(j)_oct", "gas fac(j)_nov", "gas fac(j)_dec"
# ]

# write_column_headers(worksheet, column_headers)

# # Close the workbook
# workbook.close()

import string

# def write_column_headers(worksheet, headers):
#     """
#     Write column headers to the first row of the worksheet.

#     Args:
#         worksheet: The worksheet object to write the headers to.
#         headers (list): A list of column header strings.
#     """
#     for col, header in enumerate(headers, start=1):
#         column_letter = string.ascii_uppercase[col]+"1"
#         worksheet.write(f"{column_letter}1", header)

def write_column_headers(worksheet, headers):
    """
    Write column headers to the first row of the worksheet.

    Args:
        worksheet: The worksheet object to write the headers to.
        headers (list): A list of column header strings.
    """
    for col, header in enumerate(headers, start=1):
        col_letter = chr((col - 1) % 26 + ord('A'))  # Convert column index to corresponding letter
        col_number = ((col - 1) // 26) + 1  # Calculate the column number (e.g., 1, 2, 3, ...)
        column_label = f"{col_letter}{col_number}"  # Concatenate letter and number
        worksheet.write(column_label, header)

# def write_column_headers(worksheet, headers):
#   """
#   Writes a list of headers to the first row of the given worksheet.
#   """
#   for col_index, header in enumerate(headers):
#     cell_ref = xlsxwriter.utility.xl_col_to_name(col_index) + "1"
#     worksheet.write(cell_ref, header)

# Example usage
worksheet = workbook.add_worksheet()
column_headers = [
    'total site energy', "heating", "cooling", "fans", "pumps", " Heating Gas",
    "elec fac(j)_jan", "elec fac(j)_feb", "elec fac(j)_mar", "elec fac(j)_apr",
    "elec fac(j)_may", "elec fac(j)_jun", "elec fac(j)_jul", "elec fac(j)_aug",
    "elec fac(j)_sep", "elec fac(j)_oct", "elec fac(j)_nov", "elec fac(j)_dec",
    "gas fac(j)_jan", "gas fac(j)_feb", "gas fac(j)_mar", "gas fac(j)_apr",
    "gas fac(j)_may", "gas fac(j)_jun", "gas fac(j)_jul", "gas fac(j)_aug",
    "gas fac(j)_sep", "gas fac(j)_oct", "gas fac(j)_nov", "gas fac(j)_dec"
]

write_column_headers(worksheet, column_headers)
workbook.close()