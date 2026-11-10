def parse_course(text, retries=3):
    prompt = f"""Extract information from the course description below. Output only JSON, with no other text. Format:
{{"course_name": str, "instructor": str, "time": str, "credits": int}}
Use null for any field that cannot be found.

Course description: {text}"""
    for _ in range(retries):          # retry if parsing fails (sampling gives a different result each time)
        try:
            return extract_json(chat([{"role": "user", "content": prompt}]))
        except Exception as e:
            print("Parsing failed, retrying:", e)
    return None

for t in ["Data Science Software and Computational Experiments, taught by Prof. Wei, Tuesdays 9:00-12:00, 3 credits.",
          "This semester we offer \"Introduction to Deep Learning\" (2 credits); instructor: Wang Xiaoming; meets Thursday afternoons from two to four",
          "ML w/ Prof. Lee -- Fri 10am, 3cr"]:
    print(parse_course(t))
