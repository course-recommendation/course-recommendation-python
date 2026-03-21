import random
import string

def generate_bulk_sql_insert(
    num_rows=50, 
    course_range=(794, 972), 
    attributes=None, 
    output_file="output/insert_ratings.sql"
):
    if attributes is None:
        attributes = ['theory', 'instructor', 'homework', 'material', 'exam', 'workload', 'difficulty']

    values_list = []
    
    for _ in range(num_rows):
        # Generate random user_id
        user_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        
        # Pick random course_id and attribute
        course_id = random.randint(course_range[0], course_range[1])
        attr = random.choice(attributes)
        
        # Score 1-5
        score = random.randint(1, 5)
        
        # Format this row as a SQL value tuple
        # Note: Strings are escaped with single quotes
        value_tuple = f"('{user_id}', {course_id}, '{attr}', {score})"
        values_list.append(value_tuple)

    # Combine into one command
    base_query = "INSERT INTO user_course_rating (user_id, course_id, attribute_value, score) VALUES\n"
    final_query = base_query + ",\n".join(values_list) + ";"

    # Write to file
    with open(output_file, "w") as f:
        f.write(final_query)
    
    print(f"Successfully generated 1 bulk insert statement with {num_rows} rows in {output_file}")

# Example Usage:
generate_bulk_sql_insert(
    num_rows=5000, 
    course_range=(794, 972), 
    attributes=['theory', 'instructor', 'homework', 'material', 'exam', 'workload', 'difficulty']
)