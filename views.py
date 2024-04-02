from django.shortcuts import render

from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser 
from rest_framework import status
 
from invoice.models import Invoice
from invoice.serializers import InvoiceSerializer
from rest_framework.decorators import api_view
from rest_framework.decorators import api_view, permission_classes
from rest_framework import permissions


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import HumanMessage, SystemMessage
import os
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
import json


openai_api_key = ''
os.environ["OPENAI_API_KEY"] = ''

fine_tuned_model = ChatOpenAI(
    temperature=0, model_name="gpt-4"
)


@swagger_auto_schema(method='post', manual_parameters=[
    openapi.Parameter('title', openapi.IN_FORM, description="Title of the PDF file", type=openapi.TYPE_STRING),
    openapi.Parameter('pdf_file', openapi.IN_FORM, description="PDF file to upload", type=openapi.TYPE_FILE),
])
@api_view(['POST'])
@parser_classes([MultiPartParser])
@permission_classes((permissions.AllowAny,))
def upload_invoice(request):
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        uploaded_file = request.FILES['pdf_file']
        file_path = os.path.join( 'SOURCE_DOCUMENTS', uploaded_file.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        loader = PyPDFLoader(file_path)    
        pages = loader.load_and_split() 
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

        # Choose any query of your choice
        query = "I need the following data identified correctly: Company name, Company address, Sales tax ID, Bank details, Invoice Due date, Invoice number,Product Price, Tax in %, Total price, Bezeichnung name. Give me the result like 'Company name: blablabla\n Company address:blablabla\n ...'"
        docs = docsearch.get_relevant_documents(query)

        chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
        invoice_info = chain.run(input_documents=docs, question=query)
        # invoice_info = "Company name: Grover Group GmbH\nCompany address: Holzmarktstr. 11, 10179 Berlin, Germany\nSales tax ID: DE300852104\nBank details: \n- IBAN: DE17 1004 0000 0277 7365 06\n- BIC: COBADEFFXXX\nInvoice date: 28.03.2022\nInvoice number: R572332523-S-0000503244-07M\nProduct description:\n- Apple Tablet Apple 12.9\" iPad Pro Wi-Fi + LTE 128GB (2020) (Silver)\n- Quantity: 1\n- Monthly price: 39,90 €\n- Tax rate: 19%\n- Total price: 39,90 € (including 6,37 € VAT)\nBezeichnung name: Not provided in the given context\nPosting account name: Not provided in the given contex"
        

        messages = [
            SystemMessage(
                content="You are a helpful assistant about booking account"
            ),
            HumanMessage(
                content="Which expense account to be used for these info: "+ invoice_info+ "? I only need the names." 
            ),
        ]
        invoice_account = fine_tuned_model(messages).content
        # invoice_account = "Based on the information provided, it's not possible to determine the exact expense account to be used as it depends on the company's specific chart of accounts. However, typically, the purchase of an iPad for business use could be categorized under one of the following expense accounts:\n\n1. Office Supplies Expense\n2. Computer Equipment Expense\n3. Technology Expense\n\nPlease consult with your company's accountant or financial advisor to determine the most appropriate account for this expense."

        # Split the invoice_info into lines and extract key-value pairs
        lines = invoice_info.split('\n')
        invoice_data = {}
        current_key = None

        for line in lines:
            if ':' in line:
                current_key, value = line.split(':', 1)
                invoice_data[current_key.strip()] = value.strip()
            elif current_key:
                invoice_data[current_key] += ' ' + line.strip()
        invoice_data['Posting account names'] = invoice_account
        # Convert to JSON
        json_data = json.dumps(invoice_data, indent=2)

        # Write to a JSON file
        with open('invoice_data.json', 'w') as json_file:
            json_file.write(json_data)
            
        # invoice = Invoice(title = text.title, ingredients = text.ingredients, directions = text.directions, embedding=[0.1, 0.2, 0.3])
        # invoice.save()

        return JsonResponse(json_data, safe=False)
    else:
        return JsonResponse({'message': 'Invoice upload failed!'})

@api_view(['GET'])
@permission_classes((permissions.AllowAny,))
def invoice_list(request):
    if request.method == 'GET':
        data = Invoice.objects.all()
        
        title = request.query_params.get('title', None)
        if title is not None:
            data = data.filter(title__icontains=title)
        
        invoice_serializer = InvoiceSerializer(data, many=True, context={'request': request})
        return JsonResponse(invoice_serializer.data, safe=False)
   
@api_view(['GET', 'DELETE'])
@permission_classes((permissions.AllowAny,))
def invoice_detail(request, pk):
    try: 
        invoice = Invoice.objects.get(pk=pk) 
    except Invoice.DoesNotExist: 
        return JsonResponse({'message': 'The invoice does not exist'}, status=status.HTTP_404_NOT_FOUND) 
 
    if request.method == 'GET': 
        invoice_serializer = InvoiceSerializer(invoice, context={'request': request}) 




        return JsonResponse(invoice_serializer.data) 
  
    elif request.method == 'DELETE': 
        invoice.delete() 
        return JsonResponse({'message': 'Invoice was deleted successfully!'}, status=status.HTTP_204_NO_CONTENT)