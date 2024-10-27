package com.ssau2c.spring.backend.service.impl;

import com.ssau2c.spring.backend.model.Document;
import com.ssau2c.spring.backend.model.Text;
import com.ssau2c.spring.backend.repository.DataBaseMemoryDocumentDAO;
import com.ssau2c.spring.backend.service.DocumentService;
import lombok.AllArgsConstructor;

import java.util.List;

@AllArgsConstructor
public class DataBaseDocumentServiceImpl implements DocumentService {
    private final DataBaseMemoryDocumentDAO repository;

    @Override
    public List<Document> findAllDocument() {
        return repository.findAllDocument();
    }

    @Override
    public void saveDocument(String texts) {
        repository.saveDocument(texts);
    }

    @Override
    public Document findDocumentById(int id) {
        return repository.findDocumentById(id);
    }

    @Override
    public Document updateDocument(Document document) {
        return repository.updateDocument(document);
    }

    @Override
    public void deleteDocument(int id) {
        repository.deleteDocument(id);
    }
}
