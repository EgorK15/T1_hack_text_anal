package com.ssau2c.spring.backend.repository;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssau2c.spring.backend.model.Document;
import com.ssau2c.spring.backend.model.Text;
import java.sql.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataBaseMemoryDocumentDAO {
    private Connection connection;

    public DataBaseMemoryDocumentDAO() {
        try {
            Class.forName("org.sqlite.JDBC");
            connection = DriverManager.getConnection("jdbc:sqlite:sqlite/documents.db");
        } catch (ClassNotFoundException | SQLException e) {
            System.out.println(e.getMessage());
        }
    }

    public List<Document> findAllDocument() {
        List<Document> documents = new ArrayList<>();
        try (Statement statement = connection.createStatement()) {
            String query = "SELECT * FROM document";
            ResultSet resultSet = statement.executeQuery(query);
            ObjectMapper mapper = new ObjectMapper();
            while (resultSet.next()) {
                String string = resultSet.getString("texts").replaceAll("\\n", "");
                List<Text> list = Arrays.asList(mapper.readValue(string, Text[].class));
                documents.add(
                        new Document(
                        resultSet.getInt("id"),
                        list)
                );
            }
        }
        catch (SQLException | JsonProcessingException e) {
            e.printStackTrace();
        }

        return documents;
    }

    public void saveDocument(String texts) {
        try (Statement statement = connection.createStatement()) {
            String query = "INSERT INTO document (texts) VALUES ('" + texts + "');";
            statement.execute(query);
        }
        catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public Document findDocumentById(int id) {
        Document document = null;
        try (Statement statement = connection.createStatement()) {
            String query = "SELECT * FROM document WHERE id = " + id + ";";
            ResultSet resultSet = statement.executeQuery(query);
            ObjectMapper mapper = new ObjectMapper();
            String string = resultSet.getString("texts").replaceAll("\\n", "");
            List<Text> list = Arrays.asList(mapper.readValue(string, Text[].class));
            document = new Document(
                    resultSet.getInt("id"),
                    list
            );
        }
        catch (SQLException | JsonProcessingException e) {
            e.printStackTrace();
        }
        return document;
    }

    public Document updateDocument(Document document) {
        try (Statement statement = connection.createStatement()) {
            String query = "UPDATE document SET texts = '" + document.getText() + "' WHERE id = " + document.getId();
            statement.execute(query);
        }
        catch (SQLException e) {
            e.printStackTrace();
        }
        return document;
    }

    public void deleteDocument(int id) {
        try (Statement statement = connection.createStatement()) {
            String query = "DELETE FROM document WHERE id = " + id + ";";
            statement.execute(query);
        }
        catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
