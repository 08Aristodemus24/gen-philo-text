from utilities.preprocessors import init_sequences

if __name__ == "__main__":
    try:
        init_sequences()
    except ValueError as e:
        print("You have entered a high value equal to low value e.g. np.random.randint(0, 0). Try another value")