class CharTokenizer:

    def __init__(self, text=None, characters=None):

        if characters is not None:
            self.characters = characters

        elif text is not None:
            self.characters = sorted(list(set(text)))

        else:
            raise ValueError(
                "Must provide either text or characters"
            )

        self.char_to_id = {
            char: i
            for i, char in enumerate(self.characters)
        }

        self.id_to_char = {
            i: char
            for i, char in enumerate(self.characters)
        }

    def encode(self, text):
        return [
            self.char_to_id[char]
            for char in text
        ]

    def decode(self, ids):
        return "".join(
            self.id_to_char[i]
            for i in ids
        )