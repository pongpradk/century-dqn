import time

index = []

try:
    for i in range(100):
        print(i)
        index.append(i)
        time.sleep(1)
except KeyboardInterrupt:
        print("\nTraining interrupted manually by user.")
finally:
    print("Training finished")
    print(index)