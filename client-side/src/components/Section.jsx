export default function Section({ 'section-name': name, 'children': children }){
    return (
        <section id={`${name}-section`}>
            <div className={`${name}-content`}>
                {children}
            </div>
        </section>
    );
}