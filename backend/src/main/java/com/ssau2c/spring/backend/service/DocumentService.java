package com.ssau2c.spring.backend.service;

import com.ssau2c.spring.backend.model.Document;
import com.ssau2c.spring.backend.model.Text;

import java.util.List;

public interface DocumentService {
    List<Document> findAllDocument();

    void saveDocument(String texts);

    Document findDocumentById(int id);

    Document updateDocument(Document document);

    void deleteDocument(int id);
}
