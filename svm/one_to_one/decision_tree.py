class DBT_Node:
    def __init__(self, genre1, genre2, classifier):
        self.genre1 = genre1
        self.genre2 = genre2
        self.classifier = classifier

        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def classify(self, input):
        pred_genre = self.classifier.bin_classify(input)

        if self.is_leaf():
            return pred_genre
        else:
            if pred_genre == self.genre1:
                return self.left_child.classify(input)
            else:
                return self.right_child.classify(input)


class DecisionBinaryTree:
    def __init__(self, genres, genre_classifiers):
        curr_classifier = DecisionBinaryTree.find_classifier(genres[0], genres[1], genre_classifiers)

        self.root = DBT_Node(genres[0], genres[1], curr_classifier)

        self.build_tree(genres, genre_classifiers, self.root)

    def classify(self, input):
        return self.root.classify(input)

    def print_tree(self):
        DecisionBinaryTree.print_tree_util(self.root, 0)

    # Utils
    """
    Recursively constructs the binary decision tree
    """

    def build_tree(self, genres, genre_classifiers, root):
        if len(genres) == 2:
            return

        # Remove the genre that has already been discarded if the left choice is made
        left_genres = genres.copy()
        left_genres.remove(root.genre2)
        # Remove its classifiers
        left_classifiers = DecisionBinaryTree.find_all_classifiers_except(root.genre2, genre_classifiers)

        # Find the classifier for the left child and create the node
        curr_classifier = DecisionBinaryTree.find_classifier(root.genre1, left_genres[1], genre_classifiers)
        root.left_child = DBT_Node(root.genre1, left_genres[1], curr_classifier)

        # Continue building the tree with the left child
        self.build_tree(left_genres, left_classifiers, root.left_child)

        # Remove the genre that has already been discarded if the left choice is made
        right_genres = genres.copy()
        right_genres.remove(root.genre1)
        # Remove its classifiers
        right_classifiers = DecisionBinaryTree.find_all_classifiers_except(root.genre1, genre_classifiers)

        # Find the classifier for the left child and create the node
        curr_classifier = DecisionBinaryTree.find_classifier(root.genre2, right_genres[1], genre_classifiers)
        root.right_child = DBT_Node(root.genre2, right_genres[1], curr_classifier)

        # Continue building the tree with the left child
        self.build_tree(right_genres, right_classifiers, root.right_child)

    """
    Prints a binary tree vertically
    """

    @staticmethod
    def print_tree_util(root, space):

        # Base case
        if (root == None):
            return

        # Increase distance between levels
        space += 5

        # Process right child first
        DecisionBinaryTree.print_tree_util(root.right_child, space)

        # Print current node after space
        # count
        print()
        for i in range(5, space):
            print(end=" ")
        print("(" + root.genre1 + " / " + root.genre2 + ")")

        # Process left child
        DecisionBinaryTree.print_tree_util(root.left_child, space)

    """
    Finds the binary classifier between gen_a and gen_b
    """

    @staticmethod
    def find_classifier(gen_a, gen_b, classifiers):
        buffer = None

        for classifier in classifiers:
            if (classifier.genre1 == gen_a and classifier.genre2 == gen_b) or (
                    classifier.genre1 == gen_b and classifier.genre2 == gen_a):
                buffer = classifier
                break

        # There should be a classifier for each couple
        if buffer == None:
            raise KeyError('No classifier was found for {} and {}'.format(gen_a, gen_b))

        return buffer

    """
    Finds all binary classifiers with specified genre
    """

    @staticmethod
    def find_all_classifier(genre, classifiers):
        found = list()

        for classifier in classifiers:
            if (classifier.genre1 == genre or classifier.genre2 == genre):
                found.append(classifier)

        return found

    """
    Finds all binary classifiers that do not check the excluded genre
    """

    @staticmethod
    def find_all_classifiers_except(exclude, classifiers):
        found = list()

        for classifier in classifiers:
            if classifier.genre1 != exclude and classifier.genre2 != exclude:
                found.append(classifier)

        return found