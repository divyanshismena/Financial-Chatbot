import fitz
import numpy as np
from sklearn.cluster import DBSCAN
from functools import cmp_to_key
from PIL import Image
from llama_parse import LlamaParse
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")

zoom_x = 2.0  
zoom_y = 2.0 
mat = fitz.Matrix(zoom_x, zoom_y)

class Extraction:
    def __init__(self, strategy='fast', eps=10):
        self.strategy = strategy
        self.eps = eps

    def image_scale_box(self, input_box,page,img_cop):
        x_fac = page.mediabox.width/img_cop.shape[1]
        x_fac=1/x_fac
        y_fac = page.mediabox.height/img_cop.shape[0]
        y_fac=1/y_fac
        new=[x_fac*input_box[0],y_fac*input_box[1],x_fac*input_box[2],y_fac*input_box[3]]
        return new

    def get_ocr_fitz(self, page,img_cop):
        textPage_class=page.get_textpage()
        complete=textPage_class.extractWORDS()
        box_with_words={}
        for i in complete:
            bbox=self.image_scale_box(i,page,img_cop)
            box_with_words[tuple([int(b) for b in bbox])] = i[4]
        
        return box_with_words
    
    def cluster_row(self, boxes):
        num_vertices = len(boxes)
        box_midpoints = np.zeros((num_vertices,2))
        box_midpoints[:,0] = (boxes[:,0] + boxes[:,2])/2.0
        box_midpoints[:,1] = (boxes[:,1] + boxes[:,3])/2.0
        y_coord = np.expand_dims(box_midpoints[:,1],axis=1)
        kmeans_row = DBSCAN(eps=self.eps,min_samples=1).fit(y_coord)
        rows = kmeans_row.labels_
        rows = [int(r) for r in rows]
        return rows
    
    def order_boxes(self, ocr_dic):
        boxes = np.array(list(ocr_dic.keys()))
        def box_sort(P1, P2):
            x1 = 0
            x2 = 0 
            for x in P1:
                x1 += x[1]+x[3]
            for x in P2:
                x2 += x[1]+x[3]
            x1 /= (2.0 * len(P1))
            x2 /= (2.0 * len(P2)) 

            if x1 < x2:
                return -1
            else:
                return 1

        row_ids = self.cluster_row(boxes)
        no_of_rows = len(set(row_ids))

        rows = [set([]) for _ in range(no_of_rows)]
        for i in range(no_of_rows):
            for j in range(len(boxes)):
                if row_ids[j] == i:
                    rows[i].add(tuple(boxes[j]))

        rows = sorted(rows, key=cmp_to_key(box_sort))
        for i in range(len(boxes)):
            for j in range(len(rows)):
                if tuple(boxes[i]) in rows[j]:
                    row_ids[i] = j
                    break
        
        rows_with_word = []
        
        for i, box in enumerate(boxes):
            rows_with_word.append((row_ids[i], [int(b) for b in box], ocr_dic[tuple(box)]))

        return rows_with_word

    def write_in_string(self, rows_with_word):
        text = ""
        r = 0
        for rc in rows_with_word:
            if rc[0]!=r:
                r=rc[0]
                text += '\n'+rc[-1]+' '
            else:
                text += rc[-1]+' '
        
        return text
    
    def get_text_from_page(self, page):
        map = page.get_pixmap(matrix=mat)
        map.save('bb.png')
        image = np.array(Image.open('bb.png').convert('L').convert('RGB'))
        box_with_words = self.get_ocr_fitz(page, image)
        rows_with_word = self.order_boxes(box_with_words)
        text = self.write_in_string(rows_with_word)
        return text

    def get_text_fast(self, file_path):
        document = fitz.open(file_path)
        docs, metas = [], []
        for pg, page in enumerate(document):
            text = self.get_text_from_page(page)
            metadata = {'source': os.path.basename(file_path), 'page_no': pg}
            docs.append(text)
            metas.append(metadata)

        return docs, metas

    def get_text_unstructured(self, file_path):
        pass

    def get_text_llama_parse(self, file_path):
        docs, metas = [], []
        documents = LlamaParse(result_type="markdown").get_json_result(file_path)
        for page in documents[0]['pages']:
            docs.append(page['md'])
            metas.append({'source': os.path.basename(file_path), 'page_no':page['page']})

        return docs, metas
    
    def get_text(self, files):
        texts, metas = [], []
        for file in files:
            if self.strategy == 'fast':
                text, meta = self.get_text_fast(file)
                texts.append(text)
                metas.append(meta)
            elif self.strategy == 'unst':
                text, meta = self.get_text_unstructured(file)
                texts.append(text)
                metas.append(meta)
            elif self.strategy == 'llama':
                text, meta = self.get_text_llama_parse(file)
                texts.append(text)
                metas.append(meta)

        return texts, metas