#!/usr/bin/env python3
from rag_chain import create_rag_chain
import sys


def main():
    print("free langchain rag chatbot")
    print("="*40)

    try:
        rag_chain = create_rag_chain()
    except FileNotFoundError as e:
        print(f"Error:{e}")
        return
    
    print("\n ask questions about your documents")
    print("type 'quite' or 'exit' to stop\n")

    while True:
        try:
            question = input("Question:").strip()

            if question.lower() in ['quit','exit','q']:
                print("goodbye")
                break
            
            if not question:
                continue
            
            result = rag_chain.query(question)



            print(f'answer')
            print(result["answer"])


            if result["sources"]:
                print(f"\nsources:")
                for sources in result["sources"]:
                    print(f"{sources}")

            print("\n"+"-"*50+"\n")

        except KeyboardInterrupt:
            print("goodbye")
            break
        except Exception as e:
            print(f"error :{e}")
            continue


if __name__ == "__main__":
    main()

