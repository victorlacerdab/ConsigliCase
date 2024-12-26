from utils import config_dict, get_device
from vlm import GenerationModel, Retriever

device = get_device()

retrieval = Retriever(device, config_dict=config_dict)
generator = GenerationModel(retrieval, config_dict=config_dict)

def main():
    print('\n \n \n')
    print("Welcome to ConsigliChat.")
    print("Type 'help' for available commands.")
    print(f'Make sure to encase the arguments to your commands with square brackets, e.g.: \'-i [This is a prompt.]\'.')

    while True:
        command = input("> ").strip()
        
        if not command:
            continue
        
        parts = command.split(' ')
        cmd = parts[0]
        args = ' '.join(parts[1:])
        
        if cmd == 'help':
            print("Available commands:")
            print("- -i [prompt]")
            print("- -rag [prompt]")
            print("- -frag [prompt]")
            print("- reset_chat")
            print("- regenerate_answer")
            print("- exit")
        
        elif cmd == 'interact':
            if args:
                answer = generator.interact(" ".join(args))
                print(f'{answer}')
            else:
                print("Usage: interact [prompt]")

        elif cmd == '-i':
            if args:
                for chunk in generator.stream_interact(args):
                    print(chunk, end='', flush=True)
                print()
            else:
                print("Usage: interact [prompt]")

        elif cmd == '-rag':
            try:
                cleaned_args = args.replace('[', '').replace(']', '')
                prompt = ' '.join(cleaned_args.split(' ')[:-1])

                for chunk in generator.stream_rag_interact(prompt):
                    print(chunk, end='', flush=True)
                print()
            except:
                print("Usage: -rag [prompt]")
                pass

        elif cmd == '-frag':
            try:
                cleaned_args = args.replace('[', '').replace(']', '')
                args_list = cleaned_args.split(' ')
                prompt = ' '.join(args_list[:-1])
                top_k = int(args_list[-1]) if len(args_list) > 1 else 3

                for chunk in generator.full_blown_rag(prompt, top_k):
                    print(chunk, end='', flush=True)
                print()
                
            except:
                print("Usage: -rag [prompt]")
                pass
    
        elif cmd == 'reset_chat':
            generator.chat_history = generator._init_chat_history()
            print('The session has been reset.')

        elif cmd == 'regen':
            for chunk in generator.regenerate_answer():
                print(chunk, end='', flush=True)
            print()
        
        elif cmd.lower() == 'exit':
            break
        
        else:
            print("Invalid command. Type 'help' for available commands.")

if __name__ == "__main__":
    main()