from typing import Iterator
from rag_qa.document_loaders.finance_ocr import get_ocr, recognize_finance_image
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class OCRIMGLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, img_path: str) -> None:
        """?????????????
        
                ???
                    img_path: ?????????
        
        params:
            img_path: ?????
        return:
            ??"""
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """?????????????
        
                When you're implementing lazy load methods, you should use a generator
                to yield documents one by one.
        
        params:
            ??
        return:
            ??????"""

        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        """?? img2text ???
        
        params:
            ??
        
        return:
            ??????"""
        resp = ""
        ocr = get_ocr()
        return recognize_finance_image(ocr, self.img_path)


if __name__ == '__main__':
    img_loader = OCRIMGLoader(img_path='/0001.项目目录/others/test/人工智能就业课课程大纲.png')
    doc = img_loader.load()
    print(doc)
