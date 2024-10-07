import React, { useState, useEffect } from 'react';
import { Table, Spin, Button } from 'antd';
import axios from 'axios';

interface Produit {
  id: number;
  titre_fr: string;
  sous_titre_fr: string;
  annee_edition: string;
  prix_reference_ca: string;
  resume_fr: string;
  duree: string;
  cover_image_cdn: string | null;
}

const DatabasePage: React.FC = () => {
  const [data, setData] = useState<Produit[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const apiUrl = import.meta.env.VITE_BACKEND_URL;

  useEffect(() => {
    axios.get(`${apiUrl}/produit`)
      .then(response => {
        setData(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setLoading(false);
      });
  }, []);


  const convertToCSV = (data: Produit[]) => {
    const header = Object.keys(data[0]);
    const rows = data.map(item => header.map(fieldName => JSON.stringify(item[fieldName], replacer)).join(','));
    return [header.join(','), ...rows].join('\r\n');
  };


  const replacer = (key: string, value: any) => value === null ? '' : value;


  const downloadCSV = () => {
    const BOM = '\uFEFF';
    const csvData = convertToCSV(data);
    const csvWithBOM = BOM + csvData;
    const blob = new Blob([csvWithBOM], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'produit_data.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };


  const downloadSQLiteDB = () => {
    axios.get(`${apiUrl}/export/sqlite`, { responseType: 'blob' })
      .then((response) => {
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', '1.db');  // Set the filename to '1.db'
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      })
      .catch((error) => {
        console.error('Error downloading the database:', error);
      });
  };


  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: 'Title (FR)',
      dataIndex: 'titre_fr',
      key: 'titre_fr',
    },
    {
      title: 'Subtitle (FR)',
      dataIndex: 'sous_titre_fr',
      key: 'sous_titre_fr',
    },
    {
      title: 'Year of Edition',
      dataIndex: 'annee_edition',
      key: 'annee_edition',
    },
    {
      title: 'Price (CA)',
      dataIndex: 'prix_reference_ca',
      key: 'prix_reference_ca',
    },
    {
      title: 'Duration',
      dataIndex: 'duree',
      key: 'duree',
    },
    {
      title: 'Summary (FR)',
      dataIndex: 'resume_fr',
      key: 'resume_fr',
      render: (text: string) => <div dangerouslySetInnerHTML={{ __html: text }} />,
    },
    {
      title: 'Cover Image',
      dataIndex: 'cover_image_cdn',
      key: 'cover_image_cdn',
      render: (text: string | null) => text ? <img src={text} alt="Cover" style={{ width: '100px' }} /> : 'No Image',
    },
  ];


  return (
    <div style={{ padding: '20px' }}>
      <Button 
        type="primary" 
        onClick={downloadCSV} 
        style={{ marginBottom: '20px' }}
      >
        Download CSV
      </Button>
      <Button 
        type="primary" 
        onClick={downloadSQLiteDB} 
        style={{ marginBottom: '20px', marginLeft: '10px' }}
      >
        Download SQLite DB
      </Button>
      {loading ? (
        <Spin size="large" />
      ) : (
        <Table dataSource={data} columns={columns} rowKey="id" />
      )}
    </div>
  );
};

export default DatabasePage;


