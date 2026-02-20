from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()


model1 = ChatOpenAI(model='gpt-4o-mini')
model2 = ChatOpenAI(model='gpt-4.1-mini')

prompt1 = PromptTemplate(
    template=' generate  short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = 'merge the  provided notes and quiz into a  signle document \n notes ->{notes} and quiz->{quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain  =  RunnableParallel({
    'notes'  : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser,
}
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text  = """
Key Aspects of SVM
Support Vectors: The data points closest to the hyperplane, which determine its position and orientation.
Optimal Hyperplane: The boundary line (or plane/hyperplane) that provides the maximum margin or distance between data points of different classes.
Kernel Trick: A method to map input data into higher-dimensional spaces, allowing for the classification of non-linearly separable data.
Applications: Used for classification (SVC), regression (SVR), and outlier detection, including tasks like handwritten digit recognition, face detection, and bioinformatics.
Strengths & Weaknesses: SVMs generalize well and are memory-efficient, but they are not suitable for large datasets due to high training times and require careful normalization of features. 
Wikipedia
Wikipedia
 +8
Common SVM Kernels
Linear: Used when data is linearly separable.
Polynomial: Represents the similarity of training samples in a feature space over polynomials of the original variables.
Radial Basis Function (RBF): A popular kernel for nonlinear data that maps input space into infinite-dimensional space. 
Wikipedia
Wikipedia
 +4
SVM vs. Other Algorithms
VS. Random Forest: SVMs often perform better on smaller, complex datasets, while Random Forests are faster and handle larger, noisier datasets better.
Complexity: Nonlinear SVMs are less interpretable compared to linear models, as the decision boundary is transformed.
"""


result =  chain.invoke({'text': text})
print(result)