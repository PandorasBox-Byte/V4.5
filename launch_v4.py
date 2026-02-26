from core.engine_template import Engine

def main():
    engine = Engine()
    print("EvoAI V4.5 Ready. Press Ctrl+C to exit.\n")

    while True:
        user_input = input("You: ")
        response = engine.respond(user_input)
        print("EvoAI:", response)

if __name__ == "__main__":
    main()
