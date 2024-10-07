import React, { useState,  } from 'react';
import {
    DesktopOutlined,
    FileOutlined,
    PieChartOutlined,
    TeamOutlined,
    UserOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import {Layout, Menu, theme } from 'antd';
import { useNavigate } from 'react-router-dom';

const { Content, Sider } = Layout;

type MenuItem = Required<MenuProps>['items'][number];

interface AppLayoutPorps {
    children: React.ReactNode
}


function getItem(
    label: React.ReactNode,
    key: React.Key,
    icon?: React.ReactNode,
    children?: MenuItem[],
): MenuItem {
    return {
    key,
    icon,
    children,
    label,
    } as MenuItem;
}

const items: MenuItem[] = [
    getItem('Chatbot', '', <PieChartOutlined />),
    getItem('Database', 'database', <DesktopOutlined />),
    getItem('User', 'sub1', <UserOutlined />, [
    getItem('Tom', '3'),
    getItem('Bill', '4'),
    getItem('Alex', '5'),
    ]),
    getItem('Team', 'sub2', <TeamOutlined />, [getItem('Team 1', '6'), getItem('Team 2', '8')]),
    getItem('Files', '9', <FileOutlined />),
];

const AppLayout: React.FC<AppLayoutPorps> = ({children}) => {
    const [collapsed, setCollapsed] = useState(false);
    const navigate = useNavigate()

    const {
    token: { colorBgContainer, borderRadiusLG },
    } = theme.useToken();

    const handleMenuClick = (e: any) => {
        navigate("/" + e.key);
      };

    return (
    <Layout style={{ minHeight: '100vh' }}>
        <Sider collapsible collapsed={collapsed} onCollapse={(value) => setCollapsed(value)}>
        <div className="demo-logo-vertical" />
        <Menu theme="dark" defaultSelectedKeys={['1']} mode="inline" items={items}  onClick={handleMenuClick}/>
        </Sider>
        <Content style={{ height:'100vh', margin: '0 16px' }}>
            <div
            style={{
                // padding: 24,
                height: "100vh",
                background: colorBgContainer,
                borderRadius: borderRadiusLG,
            }}
            >
            {children}
            </div>
        </Content>
    </Layout>
    );
};

export default AppLayout;